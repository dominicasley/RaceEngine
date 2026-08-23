#include <algorithm>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::TelemetryFrame;
using raceengine::TelemetryRecorder;
using raceengine::telemetryToCsv;

namespace
{

TelemetryFrame frameAt(const std::uint64_t tick)
{
    auto frame = TelemetryFrame{};
    frame.tick = tick;
    frame.time = static_cast<double>(tick) / 360.0;

    return frame;
}

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

} // namespace

TEST_CASE("the ring keeps the most recent ticks", "[physics][telemetry]")
{
    auto recorder = TelemetryRecorder(4);

    REQUIRE(recorder.capacity() == 4);
    REQUIRE(recorder.size() == 0);
    REQUIRE(recorder.inOrder().empty());

    SECTION("before it wraps it is simply everything, in order")
    {
        for (auto tick = std::uint64_t{0}; tick < 3; tick++)
        {
            recorder.record(frameAt(tick));
        }

        const auto ordered = recorder.inOrder();
        REQUIRE(ordered.size() == 3);
        REQUIRE(ordered[0].tick == 0);
        REQUIRE(ordered[2].tick == 2);
    }

    SECTION("after it wraps it is the last four, oldest first")
    {
        // The ordering is the whole point of the ring: once it has wrapped, the newest frame sits
        // *before* the oldest in memory, and anything reading them in storage order gets a run that
        // jumps back in time in the middle.
        for (auto tick = std::uint64_t{0}; tick < 10; tick++)
        {
            recorder.record(frameAt(tick));
        }

        const auto ordered = recorder.inOrder();
        REQUIRE(ordered.size() == 4);
        REQUIRE(recorder.size() == 4);
        REQUIRE(ordered[0].tick == 6);
        REQUIRE(ordered[1].tick == 7);
        REQUIRE(ordered[2].tick == 8);
        REQUIRE(ordered[3].tick == 9);

        // Time must run forward across the whole run, which is the property a plot depends on.
        for (auto index = std::size_t{1}; index < ordered.size(); index++)
        {
            REQUIRE(ordered[index].time > ordered[index - 1].time);
        }
    }

    SECTION("clearing empties it without giving the capacity back")
    {
        recorder.record(frameAt(1));
        recorder.clear();

        REQUIRE(recorder.size() == 0);
        REQUIRE(recorder.capacity() == 4);
        REQUIRE(recorder.inOrder().empty());
    }

    SECTION("a recorder with no room takes frames and keeps none")
    {
        auto empty = TelemetryRecorder(0);
        empty.record(frameAt(1));

        REQUIRE(empty.size() == 0);
        REQUIRE(empty.inOrder().empty());
    }
}

TEST_CASE("the CSV carries every channel with its units", "[physics][telemetry]")
{
    auto frame = TelemetryFrame{};
    frame.time = 1.5;
    frame.tick = 540;
    frame.velocity = glm::dvec3(0.0, 0.0, 10.0);
    // +z is forward and +x is lateral in this engine, so longitudinal acceleration goes on z.
    // Distinct magnitudes on the two axes, so a swap between them fails rather than passing.
    frame.acceleration = glm::dvec3(4.903325, 0.0, 9.80665);
    frame.yaw = 3.14159265358979 / 2.0;
    // A demand and the rim angle it makes, which are two different quantities and now two different
    // columns. Half lock on a 756-degree rim is 189 degrees; the column used to carry the demand
    // times a radians conversion and printed 28.6.
    frame.steering = 0.5;
    frame.steeringWheelAngle = 0.5 * 0.5 * 13.194689145077131;
    frame.throttle = 0.5;
    frame.brake = 0.25;
    frame.gear = 3;
    // rad/s, which is what the model works in. The column says rpm and the conversion is this
    // boundary's job — it was not being done, so every engine number in a file read 9.55 times low.
    frame.engineSpeed = 440.0;
    frame.engineTorque = 312.5;
    frame.clutchTorque = -180.25;
    frame.clutchSlip = 12.5;
    frame.clutchSlipEnergy = 4820.0;

    frame.wheels[0].verticalLoad = 3400.0;
    frame.wheels[0].suspensionTravel = 0.025;
    frame.wheels[0].damperVelocity = -0.12;
    frame.wheels[0].inContact = true;
    frame.wheels[3].inContact = false;

    const auto text = telemetryToCsv({frame});
    const auto rows = lines(text);

    REQUIRE(rows.size() == 2);

    const auto header = split(rows[0], ',');
    const auto values = split(rows[1], ',');

    SECTION("every row has exactly as many fields as the header")
    {
        REQUIRE(header.size() == values.size());
        // 32 chassis, driver and driveline channels, then fourteen per corner. Was 29 and twelve
        // until the two axle ride heights, the clutch pedal and the per-corner contact-sample count
        // were added for the telemetry expansion, and thirteen until `Patch Depth Spread` joined the
        // per-corner block for the enveloping work.
        REQUIRE(header.size() == 34 + 19 * cornerCount);
    }

    SECTION("the steering column is the rim's angle, and the demand is beside it")
    {
        // **A demand is not an angle**, and this column carried one for the whole of milestone 2:
        // `input.steering` from -1 to 1, multiplied by 57.2958 and labelled degrees. That is the
        // `Engine RPM` failure exactly — a units conversion applied to a quantity that has none.
        const auto angle = std::find(header.begin(), header.end(), "Steering Angle [deg]");
        REQUIRE(angle != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(angle - header.begin())]) ==
                Catch::Approx(189.0).epsilon(1e-4));

        const auto demand = std::find(header.begin(), header.end(), "Steering Demand []");
        REQUIRE(demand != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(demand - header.begin())]) == Catch::Approx(0.5));
    }

    SECTION("the engine column is rpm, as its name has always claimed")
    {
        const auto rpm = std::find(header.begin(), header.end(), "Engine RPM [rpm]");
        REQUIRE(rpm != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(rpm - header.begin())]) ==
                Catch::Approx(4201.69).epsilon(1e-5));
    }

    SECTION("the driveline's own channels reach the file")
    {
        for (const auto& column : {std::string("Engine Torque [Nm]"), std::string("Clutch Torque [Nm]"),
                                   std::string("Clutch Slip [rad/s]"), std::string("Clutch Slip Energy [J]")})
        {
            REQUIRE(std::find(header.begin(), header.end(), column) != header.end());
        }

        const auto torque = std::find(header.begin(), header.end(), "Clutch Torque [Nm]");
        REQUIRE(std::stod(values[static_cast<std::size_t>(torque - header.begin())]) == Catch::Approx(-180.25));

        // Accumulated rather than per tick, which is what makes it the hook a thermal model reads.
        const auto energy = std::find(header.begin(), header.end(), "Clutch Slip Energy [J]");
        REQUIRE(std::stod(values[static_cast<std::size_t>(energy - header.begin())]) == Catch::Approx(4820.0));
    }

    SECTION("each corner is named and named once")
    {
        for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
        {
            const auto tag = std::string(" ") + cornerAbbreviation(static_cast<Corner>(corner)) + " ";
            const auto count = std::count_if(header.begin(), header.end(), [&tag](const std::string& column)
                                             { return column.find(tag) != std::string::npos; });
            // Nineteen since the assist layer added five per corner — the hydraulic pressure, what
            // each of the three brake-based systems asked of the caliper, and what the wheel speed
            // sensor reported. Fourteen before that, thirteen before `Patch Depth Spread`.
            REQUIRE(count == 19);
        }
    }

    SECTION("angles are written in degrees, not the radians the model uses")
    {
        const auto yaw = std::find(header.begin(), header.end(), "Yaw [deg]");
        REQUIRE(yaw != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(yaw - header.begin())]) == Catch::Approx(90.0).epsilon(1e-4));

        // The steering column is checked in its own section above, because what makes it right is
        // not only that a radian became a degree — it is that the field being converted is an angle
        // at all. This one asserted `pi/4 -> 45` against a field holding a demand, which is a
        // conversion test passing over a quantity that had no units to convert.
    }

    SECTION("speed is km/h and acceleration is g")
    {
        const auto speed = std::find(header.begin(), header.end(), "Speed [km/h]");
        REQUIRE(speed != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(speed - header.begin())]) == Catch::Approx(36.0));

        // One g of longitudinal acceleration must read as exactly one, and the half g across the
        // car must land in the lateral column rather than in this one.
        const auto longitudinal = std::find(header.begin(), header.end(), "G Force Long [g]");
        REQUIRE(longitudinal != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(longitudinal - header.begin())]) == Catch::Approx(1.0));

        const auto lateral = std::find(header.begin(), header.end(), "G Force Lat [g]");
        REQUIRE(lateral != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(lateral - header.begin())]) == Catch::Approx(0.5));
    }

    SECTION("suspension travel is millimetres, which is how it is read")
    {
        const auto travel = std::find(header.begin(), header.end(), "Susp Pos FL [mm]");
        REQUIRE(travel != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(travel - header.begin())]) == Catch::Approx(25.0));

        const auto damper = std::find(header.begin(), header.end(), "Damper Vel FL [mm/s]");
        REQUIRE(damper != header.end());
        REQUIRE(std::stod(values[static_cast<std::size_t>(damper - header.begin())]) == Catch::Approx(-120.0));
    }

    SECTION("contact is a number, so a plot can draw it")
    {
        const auto front = std::find(header.begin(), header.end(), "In Contact FL []");
        const auto rear = std::find(header.begin(), header.end(), "In Contact RR []");
        REQUIRE(front != header.end());
        REQUIRE(rear != header.end());
        REQUIRE(values[static_cast<std::size_t>(front - header.begin())] == "1");
        REQUIRE(values[static_cast<std::size_t>(rear - header.begin())] == "0");
    }

    SECTION("the decimal separator is a point whatever the machine thinks")
    {
        // A CSV written with a comma decimal separator is not a CSV, and nothing notices until
        // somebody else opens the file. std::to_chars is what makes this true rather than a hope.
        REQUIRE(text.find(";") == std::string::npos);
        for (const auto& value : values)
        {
            REQUIRE(value.find(',') == std::string::npos);
        }
    }
}

TEST_CASE("an empty run still writes a readable file", "[physics][telemetry]")
{
    const auto text = telemetryToCsv({});
    const auto rows = lines(text);

    REQUIRE(rows.size() == 1);
    REQUIRE(rows[0].find("Time [s]") == 0);
}

TEST_CASE("a long run comes out in the order it happened", "[physics][telemetry]")
{
    // The end-to-end property: record more than the ring holds, serialise, and the file must read as
    // one run moving forward in time. Everything downstream — every plot, every acceptance check —
    // assumes this and none of it would say so if it were false.
    auto recorder = TelemetryRecorder(100);
    for (auto tick = std::uint64_t{0}; tick < 1000; tick++)
    {
        recorder.record(frameAt(tick));
    }

    const auto rows = lines(telemetryToCsv(recorder.inOrder()));
    REQUIRE(rows.size() == 101);

    auto previous = -1.0;
    for (auto index = std::size_t{1}; index < rows.size(); index++)
    {
        const auto time = std::stod(split(rows[index], ',')[0]);
        REQUIRE(time > previous);
        previous = time;
    }

    REQUIRE(previous == Catch::Approx(999.0 / 360.0));
}

TEST_CASE("each corner's columns carry that corner's own data", "[physics][telemetry][frame]")
{
    // **The corner mapping was wrong for most of milestone 2 and no test could see it.** `Susp Pos
    // FL` and `Damper Vel FL` carried the *right* corner's numbers, because `outboardSign` had the
    // two sides mirrored — in a file whose entire purpose is dropping into i2 beside a real car's.
    // It was fixed with the rest of the handedness audit, and what was left behind was a case that
    // checks the header string exists. A header is not a mapping.
    //
    // Four distinct values, one per corner, and each one asserted in its own column. A swap of any
    // two corners fails here; so does an off-by-one, and so does a transposed axle.
    auto frame = TelemetryFrame{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto marker = 1000.0 * static_cast<double>(index + 1);

        frame.wheels[index].verticalLoad = marker;
        frame.wheels[index].slipRatio = 0.001 * static_cast<double>(index + 1);
        frame.wheels[index].forceLateral = -marker;
        // Millimetres after conversion: 1, 2, 3, 4.
        frame.wheels[index].suspensionTravel = 0.001 * static_cast<double>(index + 1);
        frame.wheels[index].damperVelocity = 0.01 * static_cast<double>(index + 1);
        frame.wheels[index].inContact = index % 2 == 0;
        // Millimetres after conversion: 5, 10, 15, 20. Distinct from every other marker in this
        // fixture, so a column carrying the wrong *channel* fails as loudly as one carrying the
        // wrong corner.
        frame.wheels[index].patchDepthSpread = 0.005 * static_cast<double>(index + 1);
    }

    const auto rows = lines(telemetryToCsv({frame}));
    REQUIRE(rows.size() == 2);

    const auto header = split(rows[0], ',');
    const auto values = split(rows[1], ',');

    const auto readColumn = [&](const std::string& column)
    {
        const auto found = std::find(header.begin(), header.end(), column);
        REQUIRE(found != header.end());

        return std::stod(values[static_cast<std::size_t>(found - header.begin())]);
    };

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto tag = std::string(cornerAbbreviation(static_cast<raceengine::Corner>(index)));
        const auto expected = 1000.0 * static_cast<double>(index + 1);

        CAPTURE(tag);

        REQUIRE(readColumn("Tyre Load " + tag + " [N]") == Catch::Approx(expected));
        REQUIRE(readColumn("Tyre Force Y " + tag + " [N]") == Catch::Approx(-expected));
        REQUIRE(readColumn("Slip Ratio " + tag + " []") == Catch::Approx(0.001 * static_cast<double>(index + 1)));
        REQUIRE(readColumn("Susp Pos " + tag + " [mm]") == Catch::Approx(static_cast<double>(index + 1)));
        REQUIRE(readColumn("Damper Vel " + tag + " [mm/s]") == Catch::Approx(10.0 * static_cast<double>(index + 1)));
        REQUIRE(readColumn("In Contact " + tag + " []") == Catch::Approx(index % 2 == 0 ? 1.0 : 0.0));
        REQUIRE(readColumn("Patch Depth Spread " + tag + " [mm]") ==
                Catch::Approx(5.0 * static_cast<double>(index + 1)));
    }

    SECTION("and the four tags are the four corners, in the order the array is indexed")
    {
        // The other half of the mapping: `Corner`'s own order is what the wheel array is indexed by,
        // so a column tagged FL is corner 0 because that enumerator is 0 and for no other reason.
        REQUIRE(std::string(cornerAbbreviation(raceengine::Corner::FrontLeft)) == "FL");
        REQUIRE(std::string(cornerAbbreviation(raceengine::Corner::FrontRight)) == "FR");
        REQUIRE(std::string(cornerAbbreviation(raceengine::Corner::RearLeft)) == "RL");
        REQUIRE(std::string(cornerAbbreviation(raceengine::Corner::RearRight)) == "RR");

        REQUIRE(static_cast<std::size_t>(raceengine::Corner::FrontLeft) == 0);
        REQUIRE(static_cast<std::size_t>(raceengine::Corner::RearRight) == cornerCount - 1);
    }
}
