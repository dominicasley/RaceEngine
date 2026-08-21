#include <string>
#include <string_view>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine.physics;

using raceengine::applyVehicleTune;
using raceengine::DrivelineSetup;
using raceengine::parseVehicleTune;
using raceengine::placeholderDriveline;
using raceengine::placeholderSedan;
using raceengine::VehicleTune;

TEST_CASE("a setup file states what changed and nothing else", "[physics][setup]")
{
    const auto text = std::string_view{R"(# Bathurst, damp
front.spring   62000
front.bump      4500
front.rebound   8200
front.antiroll 22000

rear.spring    48000        # softer than it looks: the motion ratio is lower back there
diff.preload      60
diff.power_ramp    0.35
ffb.gain           0.85
)"};

    const auto tune = parseVehicleTune(text);
    REQUIRE(tune.has_value());

    SECTION("what it names arrives")
    {
        REQUIRE(tune->front.springRate == 62000.0);
        REQUIRE(tune->front.bumpRate == 4500.0);
        REQUIRE(tune->front.reboundRate == 8200.0);
        REQUIRE(tune->front.antiRollRate == 22000.0);
        REQUIRE(tune->rear.springRate == 48000.0);
        REQUIRE(tune->differential.preload == 60.0);
        REQUIRE(tune->differential.powerRamp == 0.35);
        REQUIRE(tune->feedback.gain == 0.85);
    }

    SECTION("and what it does not name stays absent rather than becoming a default")
    {
        // The difference between "the rear bar is zero" and "the file said nothing about the rear
        // bar", which is the whole reason every field is optional.
        REQUIRE_FALSE(tune->rear.antiRollRate.has_value());
        REQUIRE_FALSE(tune->rear.brakeTorque.has_value());
        REQUIRE_FALSE(tune->differential.coastRamp.has_value());
        REQUIRE_FALSE(tune->feedback.ceilingTorque.has_value());
    }
}

TEST_CASE("an empty or commented file is a valid setup that changes nothing", "[physics][setup]")
{
    for (const auto text : {"", "\n\n", "# nothing to say\n", "   \n\t\n# and a comment\n"})
    {
        const auto tune = parseVehicleTune(text);
        REQUIRE(tune.has_value());
        REQUIRE_FALSE(tune->front.springRate.has_value());
    }
}

TEST_CASE("a setup file refuses what it cannot mean", "[physics][setup]")
{
    SECTION("a key nothing is called")
    {
        // Not skipped. A misspelled key that is quietly ignored is a change the driver made, felt
        // nothing from, and then spent the afternoon compensating for somewhere else — the same
        // shape as an inert hook, which this codebase has already been caught by once.
        const auto tune = parseVehicleTune("front.sping 62000\n");
        REQUIRE_FALSE(tune.has_value());
        REQUIRE(tune.error().contains("front.sping"));
        REQUIRE(tune.error().contains("line 1"));
    }

    SECTION("an axle nothing is called")
    {
        REQUIRE_FALSE(parseVehicleTune("middle.spring 1000\n").has_value());
    }

    SECTION("a value that is not a number")
    {
        const auto tune = parseVehicleTune("front.spring stiff\n");
        REQUIRE_FALSE(tune.has_value());
        REQUIRE(tune.error().contains("not a number"));
    }

    SECTION("a number with something after it")
    {
        // '62000kg' parses as 62000 and stops, which is exactly the kind of half-read that puts a
        // plausible number on a car and no error anywhere.
        REQUIRE_FALSE(parseVehicleTune("front.spring 62000kg\n").has_value());
    }

    SECTION("a negative rate")
    {
        // A negative spring is a typo, and a car built from one drives itself into the ground in a
        // way that reads as an integrator fault four layers away.
        const auto tune = parseVehicleTune("front.spring -62000\n");
        REQUIRE_FALSE(tune.has_value());
        REQUIRE(tune.error().contains("negative"));
    }

    SECTION("a key with no value at all")
    {
        REQUIRE_FALSE(parseVehicleTune("front.spring\n").has_value());
    }

    SECTION("and the line number is in every one of them")
    {
        const auto tune = parseVehicleTune("front.spring 1000\n\n# fine so far\nrear.sping 900\n");
        REQUIRE_FALSE(tune.has_value());
        REQUIRE(tune.error().contains("line 4"));
    }
}

TEST_CASE("applying a tune changes what it names and leaves the rest of the car alone", "[physics][setup]")
{
    const auto built = placeholderSedan();
    REQUIRE(built.has_value());

    auto car = built.value();
    const auto before = built.value();

    const auto tune = parseVehicleTune("front.spring 70000\nrear.antiroll 18000\n");
    REQUIRE(tune.has_value());

    applyVehicleTune(tune.value(), car);

    SECTION("both corners of the named axle, and only that axle")
    {
        REQUIRE(car.corners[0].springRate == 70000.0);
        REQUIRE(car.corners[1].springRate == 70000.0);
        REQUIRE(car.corners[2].springRate == before.corners[2].springRate);
        REQUIRE(car.corners[3].springRate == before.corners[3].springRate);

        REQUIRE(car.corners[2].antiRollRate == 18000.0);
        REQUIRE(car.corners[3].antiRollRate == 18000.0);
        REQUIRE(car.corners[0].antiRollRate == before.corners[0].antiRollRate);
    }

    SECTION("and nothing a setup sheet has no business touching")
    {
        // The geometry is the car and the tune is a sheet clipped to it. A format that could move a
        // hardpoint would be a format in which a typo produces a different vehicle and calls it a
        // setup change.
        REQUIRE(car.corners[0].hardpoints.strutTop.x == before.corners[0].hardpoints.strutTop.x);
        REQUIRE(car.corners[0].hardpoints.strutTop.y == before.corners[0].hardpoints.strutTop.y);
        REQUIRE(car.corners[0].unsprungMass == before.corners[0].unsprungMass);
        REQUIRE(car.corners[0].tireVerticalRate == before.corners[0].tireVerticalRate);
        REQUIRE(car.sprung.size() == before.sprung.size());
    }
}

TEST_CASE("a damper stated on one side keeps what the other side already was", "[physics][setup]")
{
    const auto built = placeholderSedan();
    REQUIRE(built.has_value());

    auto car = built.value();
    // Read off the curve rather than assumed: `linearDamper` puts bump on the positive velocity
    // side, so this is where each of the two actually lives.
    const auto rebound = built->corners[0].damper.at(-1.0);

    const auto tune = parseVehicleTune("front.bump 5000\n");
    REQUIRE(tune.has_value());

    applyVehicleTune(tune.value(), car);

    // Bump changed, rebound did not — a sheet that states one of the two must be read against what
    // the other already is rather than against a default, or half of every damper change is a second
    // change nobody asked for.
    REQUIRE(car.corners[0].damper.at(1.0) == Catch::Approx(5000.0));
    REQUIRE(car.corners[0].damper.at(-1.0) == Catch::Approx(rebound));
}

TEST_CASE("the differential's own numbers arrive through the same file", "[physics][setup]")
{
    auto driveline = placeholderDriveline();
    const auto coast = driveline.differential.coastRamp;

    const auto tune = parseVehicleTune("diff.preload 90\ndiff.power_ramp 0.4\n");
    REQUIRE(tune.has_value());

    applyVehicleTune(tune.value(), driveline);

    REQUIRE(driveline.differential.preload == 90.0);
    REQUIRE(driveline.differential.powerRamp == 0.4);
    REQUIRE(driveline.differential.coastRamp == coast);
}
