// **Fifty-two per-corner columns, fifty-two assertions, and not one of them is "the header says FL".**
//
// The rack trace grew from thirteen channels to sixty-three so that the power assist's *aim* could be
// measured and not only its shape — see `VehicleTrace` — and to eighty-five when the electronics
// joined it on 2026-08-23. Thirteen of the channels are per corner, and four corners of thirteen is
// fifty-two fresh opportunities for the FL column to carry the FR corner. That fault is invisible in every plot: the
// numbers are all the right size, all the right sign, and all in the wrong place, and the first thing it costs is a
// conclusion about which end of the car is losing grip.
//
// A test that checked the header strings existed would pass against a writer that emitted the corners
// backwards, so this does the opposite: every channel of every corner is given a value that no other
// channel of any other corner has, the file is written, the header is read, and each column is
// required to hold *its own* number. A swap of any two corners fails four assertions; a swap of two
// channels within a corner fails two; a transposed loop fails most of them.
//
// The chassis and driver columns get the same treatment for the same reason, minus the corner axis.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;

using raceengine::RackTorqueFrame;
using raceengine::rackTorqueToCsv;
using raceengine::tracedCornerAbbreviations;
using raceengine::tracedCornerCount;

namespace
{

std::vector<std::string> split(const std::string& line, const char separator)
{
    auto parts = std::vector<std::string>{};
    auto start = std::size_t{0};

    while (true)
    {
        const auto found = line.find(separator, start);
        parts.push_back(line.substr(start, found - start));
        if (found == std::string::npos)
        {
            break;
        }
        start = found + 1;
    }

    return parts;
}

std::vector<std::string> lines(const std::string& text)
{
    auto out = std::vector<std::string>{};
    for (const auto& line : split(text, '\n'))
    {
        if (!line.empty())
        {
            out.push_back(line);
        }
    }

    return out;
}

// A distinct number for every (corner, channel) pair, built so that no two pairs can collide: the
// corner picks the hundreds and the channel picks the units, and the sign alternates so a corner
// swap cannot be hidden by a symmetric car. Nothing here is physical — a value that looked like a
// plausible tyre load would be one a reader could confuse with a real one.
[[nodiscard]] double sentinel(const std::size_t corner, const std::size_t channel)
{
    const auto magnitude = 1000.0 * static_cast<double>(corner + 1) + 10.0 * static_cast<double>(channel + 1);

    return (corner + channel) % 2 == 0 ? magnitude : -magnitude;
}

// The header index of a named column, required to exist. Returning the index rather than an iterator
// because every use of it is a subscript into the value row.
[[nodiscard]] std::size_t columnOf(const std::vector<std::string>& header, const std::string& name)
{
    const auto found = std::find(header.begin(), header.end(), name);

    CAPTURE(name);
    REQUIRE(found != header.end());

    return static_cast<std::size_t>(found - header.begin());
}

} // namespace

TEST_CASE("every per-corner column of the rack trace carries its own corner", "[input][ffb][trace][frame]")
{
    auto frame = RackTorqueFrame{};

    for (auto corner = std::size_t{0}; corner < tracedCornerCount; corner++)
    {
        auto& wheel = frame.vehicle.wheels[corner];

        // Radians in, degrees out, so the sentinel is divided back down here and the assertion below
        // asks for the whole number. That makes the conversion part of what is pinned rather than
        // something the test quietly agrees with.
        wheel.slipAngle = sentinel(corner, 0) / 57.29577951308232;
        wheel.slipRatio = sentinel(corner, 1);
        wheel.verticalLoad = sentinel(corner, 2);
        wheel.lateralForce = sentinel(corner, 3);
        wheel.longitudinalForce = sentinel(corner, 4);
        wheel.aligningMoment = sentinel(corner, 5);
        // Metres in, millimetres out.
        wheel.suspensionTravel = sentinel(corner, 6) / 1000.0;
        wheel.damperVelocity = sentinel(corner, 7) / 1000.0;
        // The one unsigned channel, and an integer. Distinct per corner all the same.
        wheel.contactingSamples = static_cast<std::uint32_t>(7 + corner);
        // Metres in, millimetres out. The real channel is never negative — a spread is a difference
        // between two depths on the same side of zero — but the sentinel alternates sign like every
        // other one here, because what this test is built to catch is a corner swap and the writer
        // has no opinion about the sign of what it is handed.
        wheel.patchDepthSpread = sentinel(corner, 8) / 1000.0;

        // The electronics' per-corner channels, joined 2026-08-23. Two of the three are unsigned, so
        // they are made distinct per corner by construction rather than by the alternating sentinel:
        // a swap between two corners has to be visible in every column of the block or the block is
        // not covered by this test.
        wheel.antilockActive = corner % 2 == 0;
        wheel.antilockCycles = static_cast<std::uint32_t>(31 + corner);
        // Pascals in, bar out.
        wheel.brakePressure = sentinel(corner, 9) * 1.0e5;
    }

    const auto rows = lines(rackTorqueToCsv({frame}));
    REQUIRE(rows.size() == 2);

    const auto header = split(rows[0], ',');
    const auto values = split(rows[1], ',');

    REQUIRE(header.size() == values.size());
    // Thirteen that were always there, then nine chassis channels, five driver ones, **six for the
    // electronics** and **thirteen** per corner — eighty-five in total.
    //
    // The history, because the count has been wrong in a brief before: it was nine per corner and
    // sixty-three until `Patch Depth Spread` joined the block for the enveloping work, then ten and
    // sixty-seven. On 2026-08-23 the electronics joined it — `ABS Fitted`, `TC Mode` and the three
    // active flags with the engine reduction on the chassis side, and `ABS Active`, `ABS Cycles` and
    // `Brake Pressure` per corner. That was added because a trace could not say whether it had been
    // driven with the anti-lock system on, and the setup sheet had been edited since. The telemetry
    // brief that asked for the original expansion counted forty-nine new and sixty-two, because it
    // listed nine chassis channels under a heading that said eight; the list is what was built.
    REQUIRE(header.size() == 13 + 9 + 5 + 6 + 13 * tracedCornerCount);

    for (auto corner = std::size_t{0}; corner < tracedCornerCount; corner++)
    {
        const auto tag = std::string(" ") + tracedCornerAbbreviations[corner];

        // The units are part of the name, so they are part of what is looked up: a channel whose
        // header said `[deg]` while the writer emitted radians would fail to be found here rather
        // than quietly matching on the stem.
        const auto named = [&](const std::string& channel, const std::string& units)
        {
            return std::stod(values[columnOf(header, channel + tag + " " + units)]);
        };

        CAPTURE(corner, tracedCornerAbbreviations[corner]);

        REQUIRE(named("Slip Angle", "[deg]") == Catch::Approx(sentinel(corner, 0)).epsilon(1e-6));
        REQUIRE(named("Slip Ratio", "[]") == Catch::Approx(sentinel(corner, 1)).epsilon(1e-9));
        REQUIRE(named("Tyre Fz", "[N]") == Catch::Approx(sentinel(corner, 2)).epsilon(1e-9));
        REQUIRE(named("Tyre Fy", "[N]") == Catch::Approx(sentinel(corner, 3)).epsilon(1e-9));
        REQUIRE(named("Tyre Fx", "[N]") == Catch::Approx(sentinel(corner, 4)).epsilon(1e-9));
        REQUIRE(named("Tyre Mz", "[Nm]") == Catch::Approx(sentinel(corner, 5)).epsilon(1e-9));
        REQUIRE(named("Susp Pos", "[mm]") == Catch::Approx(sentinel(corner, 6)).epsilon(1e-9));
        REQUIRE(named("Damper Vel", "[mm/s]") == Catch::Approx(sentinel(corner, 7)).epsilon(1e-9));
        REQUIRE(named("Contact Samples", "[]") == Catch::Approx(static_cast<double>(7 + corner)).margin(1e-9));
        REQUIRE(named("Patch Depth Spread", "[mm]") == Catch::Approx(sentinel(corner, 8)).epsilon(1e-9));
        REQUIRE(named("ABS Active", "[]") == Catch::Approx(corner % 2 == 0 ? 1.0 : 0.0).margin(1e-9));
        REQUIRE(named("ABS Cycles", "[]") == Catch::Approx(static_cast<double>(31 + corner)).margin(1e-9));
        REQUIRE(named("Brake Pressure", "[bar]") == Catch::Approx(sentinel(corner, 9)).epsilon(1e-9));
    }
}

TEST_CASE("and the chassis and driver columns carry what they are named", "[input][ffb][trace]")
{
    // The same discipline without the corner axis. Every value is distinct, so a pair of columns
    // swapped between two channels of the same units — `CoG X` for `CoG Z`, lateral g for
    // longitudinal — fails rather than passing because both happened to be plausible.
    auto frame = RackTorqueFrame{};
    auto& car = frame.vehicle;

    car.centreOfMassX = 123.25;
    car.centreOfMassZ = -456.75;
    // Radians in, degrees out.
    car.heading = 1.5707963267948966;
    // m/s in, km/h out.
    car.speed = 30.0;
    // m/s^2 in, g out, and deliberately different magnitudes on the two axes.
    car.lateralAcceleration = 9.80665;
    car.longitudinalAcceleration = -4.903325;
    car.yawRate = 0.5;
    car.rideHeightFront = 0.125;
    car.rideHeightRear = 0.148;
    car.throttle = 0.75;
    car.brake = 0.25;
    car.clutch = 0.5;
    car.gear = 4;
    // rad/s in, rpm out. This is the conversion whose absence survived a whole milestone in the
    // other file — every engine number read 9.55 times low — so it is pinned here from the start.
    car.engineSpeed = 440.0;

    const auto rows = lines(rackTorqueToCsv({frame}));
    REQUIRE(rows.size() == 2);

    const auto header = split(rows[0], ',');
    const auto values = split(rows[1], ',');

    const auto named = [&](const std::string& channel)
    {
        return std::stod(values[columnOf(header, channel)]);
    };

    REQUIRE(named("CoG X [m]") == Catch::Approx(123.25));
    REQUIRE(named("CoG Z [m]") == Catch::Approx(-456.75));
    REQUIRE(named("Heading [deg]") == Catch::Approx(90.0).epsilon(1e-6));
    REQUIRE(named("Speed [kph]") == Catch::Approx(108.0).epsilon(1e-6));
    REQUIRE(named("G Force Lat [g]") == Catch::Approx(1.0).epsilon(1e-5));
    REQUIRE(named("G Force Long [g]") == Catch::Approx(-0.5).epsilon(1e-5));
    REQUIRE(named("Yaw Rate [deg/s]") == Catch::Approx(28.6478897565).epsilon(1e-6));
    REQUIRE(named("Ride Height F [mm]") == Catch::Approx(125.0).epsilon(1e-9));
    REQUIRE(named("Ride Height R [mm]") == Catch::Approx(148.0).epsilon(1e-9));
    REQUIRE(named("Throttle Pos [%]") == Catch::Approx(75.0).epsilon(1e-9));
    REQUIRE(named("Brake Pos [%]") == Catch::Approx(25.0).epsilon(1e-9));
    REQUIRE(named("Clutch Pos [%]") == Catch::Approx(50.0).epsilon(1e-9));
    REQUIRE(named("Gear []") == Catch::Approx(4.0).margin(1e-9));
    REQUIRE(named("Engine RPM [rpm]") == Catch::Approx(4201.7).epsilon(1e-4));
}

TEST_CASE("and the thirteen columns that were always there have not moved", "[input][ffb][trace]")
{
    // The expansion's own regression guard. Fifty new columns were appended to a file that sessions
    // of analysis already read, and the one thing that must not have happened is that any of the
    // original thirteen changed name, position or units on the way.
    auto frame = RackTorqueFrame{};
    frame.time = 1.5;
    frame.sequence = 540;
    frame.steeringTorque = 12.5;
    frame.assistedTorque = 4.25;
    frame.rackForce = 1800.0;
    frame.tyreRackForce = 1750.0;
    // Metres in, millimetres out, for both.
    frame.rackTravel = 0.0125;
    frame.rackVelocity = -0.032;
    frame.requestedTorque = 4.25;
    frame.commandedTorque = 4.0;
    frame.deliveredTorque = 3.96;
    frame.clipped = true;
    frame.latencyMilliseconds = 2.26;

    const auto rows = lines(rackTorqueToCsv({frame}));
    REQUIRE(rows.size() == 2);

    const auto header = split(rows[0], ',');
    const auto values = split(rows[1], ',');

    // Position as well as name: the original thirteen are the first thirteen, in order, so anything
    // reading the file by column index rather than by header still reads what it always did.
    const auto original = std::vector<std::string>{"Time [s]",
                                                   "Sequence",
                                                   "Steering Torque [Nm]",
                                                   "Assisted Torque [Nm]",
                                                   "Rack Force [N]",
                                                   "Tyre Rack Force [N]",
                                                   "Rack Travel [mm]",
                                                   "Rack Vel [mm/s]",
                                                   "Requested Torque [Nm]",
                                                   "Commanded Torque [Nm]",
                                                   "Delivered Torque [Nm]",
                                                   "Clipped []",
                                                   "Latency [ms]"};

    for (auto index = std::size_t{0}; index < original.size(); index++)
    {
        CAPTURE(index, original[index]);
        REQUIRE(header[index] == original[index]);
    }

    REQUIRE(std::stod(values[0]) == Catch::Approx(1.5));
    REQUIRE(std::stod(values[1]) == Catch::Approx(540.0));
    REQUIRE(std::stod(values[2]) == Catch::Approx(12.5));
    REQUIRE(std::stod(values[3]) == Catch::Approx(4.25));
    REQUIRE(std::stod(values[4]) == Catch::Approx(1800.0));
    REQUIRE(std::stod(values[5]) == Catch::Approx(1750.0));
    REQUIRE(std::stod(values[6]) == Catch::Approx(12.5));
    REQUIRE(std::stod(values[7]) == Catch::Approx(-32.0));
    REQUIRE(std::stod(values[8]) == Catch::Approx(4.25));
    REQUIRE(std::stod(values[9]) == Catch::Approx(4.0));
    REQUIRE(std::stod(values[10]) == Catch::Approx(3.96));
    REQUIRE(std::stod(values[11]) == Catch::Approx(1.0));
    REQUIRE(std::stod(values[12]) == Catch::Approx(2.26));
}
