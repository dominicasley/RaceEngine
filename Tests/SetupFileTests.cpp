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

TEST_CASE("the bump stop's dissipation is the driver's to restate", "[physics][setup]")
{
    // `front.stopdamping` and `front.stophysteresis` exist because the shipped 40000 N·s/m is a
    // placed number about five times the front corner's critical damping, and the sourced
    // alternative — BASF's hysteresis-loop band, `stopdamping 0` with `stophysteresis 0.07` — is a
    // feel change that belongs on the sheet, beside `friction`, changed between two laps rather
    // than two builds.
    const auto built = placeholderSedan();
    REQUIRE(built.has_value());

    auto car = built.value();
    const auto before = built.value();

    const auto tune = parseVehicleTune("front.stopdamping 0\nfront.stophysteresis 0.07\n");
    REQUIRE(tune.has_value());

    applyVehicleTune(tune.value(), car);

    SECTION("both front corners' bump stops and nothing else")
    {
        REQUIRE(car.corners[0].bumpStop.damping == 0.0);
        REQUIRE(car.corners[1].bumpStop.damping == 0.0);
        REQUIRE(car.corners[0].bumpStop.hysteresis == 0.07);
        REQUIRE(car.corners[1].bumpStop.hysteresis == 0.07);

        REQUIRE(car.corners[2].bumpStop.damping == before.corners[2].bumpStop.damping);
        REQUIRE(car.corners[2].bumpStop.hysteresis == 0.0);
    }

    SECTION("the droop stop is a different mechanism and the key does not reach it")
    {
        // On a strut the rebound limit is the damper topping out, not a bumper somebody specifies
        // (the droop-travel account in docs/known-red.md). A key named for the bump stop that
        // quietly restated both would be a change the driver did not ask for.
        REQUIRE(car.corners[0].droopStop.damping == before.corners[0].droopStop.damping);
        REQUIRE(car.corners[1].droopStop.damping == before.corners[1].droopStop.damping);
    }

    SECTION("the elastic half of the stop is the car's and stays it")
    {
        REQUIRE(car.corners[0].bumpStop.gap == before.corners[0].bumpStop.gap);
        REQUIRE(car.corners[0].bumpStop.rate == before.corners[0].bumpStop.rate);
        REQUIRE(car.corners[0].bumpStop.progression == before.corners[0].bumpStop.progression);
    }
}

TEST_CASE("compliance camber rides the sheet in the unit the rig reports", "[physics][setup]")
{
    // `compliancecamber` is degrees per kilonewton on the sheet — the unit the K&C figures behind
    // it are quoted in — and radians per newton on the car, converted exactly as `compliancesteer`
    // is. `front.compliancecamber 0` is the A/B against the Golf's own stated 0.17.
    const auto built = placeholderSedan();
    REQUIRE(built.has_value());

    auto car = built.value();
    const auto before = built.value();

    const auto tune = parseVehicleTune("front.compliancecamber 0.17\n");
    REQUIRE(tune.has_value());

    applyVehicleTune(tune.value(), car);

    REQUIRE(car.corners[0].lateralForceCamber == 0.17 * 0.017453292519943295 / 1000.0);
    REQUIRE(car.corners[1].lateralForceCamber == 0.17 * 0.017453292519943295 / 1000.0);
    REQUIRE(car.corners[2].lateralForceCamber == before.corners[2].lateralForceCamber);
    REQUIRE(car.corners[3].lateralForceCamber == before.corners[3].lateralForceCamber);

    // And the steer coefficient beside it is untouched — two keys, two mechanisms.
    REQUIRE(car.corners[0].lateralForceSteer == before.corners[0].lateralForceSteer);
}

TEST_CASE("recession rides the sheet in the unit the design band is quoted in", "[physics][setup]")
{
    // `recession` is millimetres per kilonewton on the sheet — the unit Heissing/Ersoy's front
    // band is quoted in — and metres per newton on the car. The Golf states 6 front / 10 rear
    // since 2026-08-29 night, so `recession 0` is the A/B and the way back; the worked conversion
    // is beside the Golf's compliance figures in `PublishedCarsImpl.cpp`.
    const auto built = placeholderSedan();
    REQUIRE(built.has_value());

    auto car = built.value();
    const auto before = built.value();

    const auto tune = parseVehicleTune("front.recession 6\nrear.recession 10\n");
    REQUIRE(tune.has_value());

    applyVehicleTune(tune.value(), car);

    REQUIRE(car.corners[0].longitudinalForceRecession == 6.0 * 1.0e-3 / 1000.0);
    REQUIRE(car.corners[1].longitudinalForceRecession == 6.0 * 1.0e-3 / 1000.0);
    REQUIRE(car.corners[2].longitudinalForceRecession == 10.0 * 1.0e-3 / 1000.0);
    REQUIRE(car.corners[3].longitudinalForceRecession == 10.0 * 1.0e-3 / 1000.0);

    // The two rotation coefficients beside it are untouched — three channels, three keys.
    REQUIRE(car.corners[0].lateralForceSteer == before.corners[0].lateralForceSteer);
    REQUIRE(car.corners[0].lateralForceCamber == before.corners[0].lateralForceCamber);
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

TEST_CASE("the pedal cue's thresholds are the driver's dials", "[physics][setup]")
{
    // The game applies these to the input layer's `PedalFeedbackSetup` itself — this module cannot
    // name that type — so what is pinned here is the sheet half: the keys parse, an unstated one
    // stays absent rather than becoming a default, and a sheet stating only a pedal number still
    // counts as saying something.
    const auto tune = parseVehicleTune("pedal.brake_full 2.0\npedal.throttle_full 6\n");
    REQUIRE(tune.has_value());

    REQUIRE(tune->pedal.brakeFullPeaks == 2.0);
    REQUIRE(tune->pedal.throttleFullPeaks == 6.0);
    REQUIRE_FALSE(tune->pedal.onsetPeaks.has_value());

    REQUIRE(raceengine::statesAnything(tune.value()));
    REQUIRE_FALSE(raceengine::statesAnything(parseVehicleTune("").value()));

    const auto onset = parseVehicleTune("pedal.onset 1.05\n");
    REQUIRE(onset.has_value());
    REQUIRE(onset->pedal.onsetPeaks == 1.05);
}

TEST_CASE("a setup sheet states which electronics are fitted", "[physics][setup][assists]")
{
    // The assists live on the sheet rather than in the environment, because which of them are on is
    // a driver's setting in the way `ffb.gain` is: felt, argued about, and changed between two laps.

    SECTION("absent keys leave the car with none of them, which is what the factory default here is")
    {
        const auto tune = raceengine::parseVehicleTune("front.spring 40000\n");
        REQUIRE(tune.has_value());

        REQUIRE_FALSE(tune->assists.antilock.has_value());
        REQUIRE_FALSE(tune->assists.traction.has_value());
        REQUIRE_FALSE(tune->assists.cornering.has_value());

        const auto built = raceengine::golfGtiMk7();
        REQUIRE(built.has_value());

        auto assists = raceengine::golfGtiMk7Assists(built.value());
        raceengine::applyVehicleTune(tune.value(), assists);

        REQUIRE_FALSE(assists.antilock.enabled);
        REQUIRE(assists.traction.mode == raceengine::TractionMode::Off);
        REQUIRE_FALSE(assists.cornering.enabled);
    }

    SECTION("and stated keys switch exactly what they name")
    {
        const auto tune = raceengine::parseVehicleTune("assist.abs 1\nassist.tc sport\nassist.xds 1\n");
        REQUIRE(tune.has_value());
        REQUIRE(raceengine::statesAnything(tune.value()));

        const auto built = raceengine::golfGtiMk7();
        REQUIRE(built.has_value());

        auto assists = raceengine::golfGtiMk7Assists(built.value());
        raceengine::applyVehicleTune(tune.value(), assists);

        REQUIRE(assists.antilock.enabled);
        REQUIRE(assists.traction.mode == raceengine::TractionMode::Sport);
        REQUIRE(assists.cornering.enabled);
    }

    SECTION("a mode is a name, so a number is not one")
    {
        // `assist.tc 2` would be a sheet nobody can read and a typo nobody can see.
        const auto refused = raceengine::parseVehicleTune("assist.tc 2\n");
        REQUIRE_FALSE(refused.has_value());
        REQUIRE(refused.error().find("assist.tc") != std::string::npos);

        const auto misspelled = raceengine::parseVehicleTune("assist.tc ful\n");
        REQUIRE_FALSE(misspelled.has_value());
    }

    SECTION("switching one off is a line that says so, not a line deleted")
    {
        // The overlay is applied to a freshly built car every time, so a deleted line reverts. This
        // is the other direction: an explicit `off` on a car that had it on.
        const auto tune = raceengine::parseVehicleTune("assist.tc off\n");
        REQUIRE(tune.has_value());

        const auto built = raceengine::golfGtiMk7();
        REQUIRE(built.has_value());

        auto assists = raceengine::golfGtiMk7Assists(built.value());
        assists.traction.mode = raceengine::TractionMode::Full;

        raceengine::applyVehicleTune(tune.value(), assists);

        REQUIRE(assists.traction.mode == raceengine::TractionMode::Off);
    }
}
