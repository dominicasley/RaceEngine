#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::computeRollCentre;
using raceengine::CornerHardpoints;
using raceengine::CornerSetup;
using raceengine::CornerSide;
using raceengine::golfMk7FrontCorner;
using raceengine::golfMk7RearCorner;
using raceengine::solveCorner;
using raceengine::sweepCorner;
using raceengine::validateCorner;
using raceengine::validateCornerSetup;

// Acceptance criterion 3: the suspension solver against *published* geometry rather than against
// numbers chosen to make it look good.
//
// The car is a Volkswagen Golf GTI Mk7.5, whose hardpoints are transcribed in a publicly
// distributed Assetto Corsa model. Only the suspension pickup points are here — they are
// measurements of a production car's linkage, and they are here as reference geometry for
// validating a solver, which is what they are good for.
//
// **The conversion is verified, not assumed.** Assetto Corsa states its pickup points relative to
// the wheel centre, with x measured *inboard* from the wheel's centre plane, y up from the wheel
// centre and z forward. Converting with that and reading the track back off the resulting geometry
// reproduces the track the same file states, to four decimal places, on both axles — which is the
// check, because any other convention misses it by tens of centimetres. See
// `scripts/import-ac-car.py`, which does the conversion and prints the frame-independent link
// lengths beside it.
//
//     x = trackWidth / 2 - x_ac
//     y = y_ac + tyreRadius          (so the design contact patch sits on y = 0)
//     z = z_ac + axleStation

namespace
{

// The geometry itself lives in `Physics/Api/PublishedCars.cppm`, converted there from the file's own
// pickup points, so that the car the engine builds and the car these curves are read off cannot come
// apart. Front is a MacPherson strut on a 1.539 m track, rear a double wishbone on 1.516 m, both with
// a 0.3298 m tyre and their axles at plus and minus half the 2.638 m wheelbase.
//
// A production Mk7.5 has a multi-link rear rather than a wishbone; the model approximates it as one,
// which is why the lower arm's pivot axis is 0.74 m long — that is a trailing arm wearing a
// wishbone's clothes. The approximation is a reasonable one and the curves it produces are a road
// car's, but the geometry is a model of a model and it is worth knowing that.
CornerHardpoints golfFront()
{
    return golfMk7FrontCorner(CornerSide::Right);
}

CornerHardpoints golfRear()
{
    // The engine's rear corner carries damper mounts, which AC does not state and which the last case
    // in this file is about. Stripped back off, this is the file's geometry and nothing else.
    auto corner = golfMk7RearCorner(CornerSide::Right);
    corner.damperChassis = glm::dvec3(0.0);
    corner.damperWishbone = glm::dvec3(0.0);

    return corner;
}

constexpr auto degrees = 57.29577951308232;

// The gradient of a channel about the static position, per ten millimetres of bump.
template <typename Read> double perTenMillimetres(const raceengine::SuspensionSweep& sweep, Read read)
{
    const auto middle = sweep.samples.size() / 2;
    const auto& below = sweep.samples[middle - 1];
    const auto& above = sweep.samples[middle + 1];

    return (read(above) - read(below)) / (above.wheelTravel - below.wheelTravel) * 0.01;
}

} // namespace

TEST_CASE("the conversion from published data reproduces the track it states", "[physics][published]")
{
    // The check that says the coordinate convention is right. It is the only part of the import
    // that cannot be verified from the file's internal consistency alone, and getting it wrong is
    // not subtle — it misses by tens of centimetres.
    // A track is a width, so it is the magnitude that has to match: which side of the car +x is
    // on is `outboardSign`'s to say, and it says the car's left.
    REQUIRE(2.0 * std::abs(golfFront().wheelCentre.x) == Catch::Approx(1.539).margin(1e-4));
    REQUIRE(2.0 * std::abs(golfRear().wheelCentre.x) == Catch::Approx(1.516).margin(1e-4));
}

TEST_CASE("a production front strut solves and behaves like one", "[physics][published][strut]")
{
    const auto corner = golfFront();

    REQUIRE(validateCorner(corner).has_value());

    const auto sweep = sweepCorner(corner, 41);
    REQUIRE(sweep.has_value());

    SECTION("its kingpin is inclined the way a road strut's is")
    {
        const auto kingpin = glm::normalize(corner.strutTop - corner.lower.ballJoint);
        const auto inclination = std::acos(std::abs(kingpin.y)) * degrees;

        // Twelve to sixteen degrees is what a production strut runs.
        REQUIRE(inclination == Catch::Approx(15.5).margin(1.0));
    }

    SECTION("the strut is the damper, so the motion ratio is near one")
    {
        for (const auto& sample : sweep->samples)
        {
            REQUIRE(std::abs(sample.motionRatio) > 0.85);
            REQUIRE(std::abs(sample.motionRatio) < 1.00);
        }
    }

    SECTION("camber gain is negative in bump and about a quarter of a degree per centimetre")
    {
        const auto gain = perTenMillimetres(sweep.value(), [](const auto& s) { return s.camber * degrees; });

        REQUIRE(gain < 0.0);
        REQUIRE(gain == Catch::Approx(-0.254).margin(0.03));

        // And across the whole travel it swings from positive at droop to negative at bump.
        REQUIRE(sweep->samples.front().camber * degrees > 1.0);
        REQUIRE(sweep->samples.back().camber * degrees < -0.8);
    }

    SECTION("bump steer is small but not nothing, which is a rack-and-strut front")
    {
        const auto steer = perTenMillimetres(sweep.value(), [](const auto& s) { return s.toe * degrees; });

        REQUIRE(std::abs(steer) == Catch::Approx(0.097).margin(0.03));
    }

    SECTION("the wheel scrubs sideways as it rises, which is what a strut does")
    {
        // Half-track change, so it is one wheel's lateral movement rather than the pair's.
        const auto scrub = perTenMillimetres(sweep.value(), [](const auto& s) { return s.halfTrack * 1000.0; });

        REQUIRE(scrub == Catch::Approx(1.63).margin(0.10));
    }

    SECTION("the roll centre is above ground and falls as the car compresses")
    {
        auto atDroop = sweep->samples.front();
        auto atBump = sweep->samples.back();
        computeRollCentre(corner, atDroop);
        computeRollCentre(corner, atBump);

        REQUIRE(atDroop.rollCentreHeight > atBump.rollCentreHeight);
        REQUIRE(atBump.rollCentreHeight > 0.0);
        // A front strut's roll centre lives somewhere around a hundred millimetres.
        REQUIRE(atDroop.rollCentreHeight < 0.30);
    }

    SECTION("steering it changes its camber, because the kingpin is inclined")
    {
        const auto oneWay = solveCorner(corner, 0.0, 0.035);
        const auto theOther = solveCorner(corner, 0.0, -0.035);

        REQUIRE(oneWay.has_value());
        REQUIRE(theOther.has_value());
        REQUIRE(std::abs(oneWay->toe) > 0.15); // more than eight degrees of steer
        REQUIRE(oneWay->camber * theOther->camber < 0.0);
    }
}

TEST_CASE("a production rear wishbone solves and behaves like one", "[physics][published]")
{
    const auto corner = golfRear();

    REQUIRE(validateCorner(corner).has_value());

    const auto sweep = sweepCorner(corner, 41);
    REQUIRE(sweep.has_value());

    SECTION("camber gain is negative in bump and gentler than the front's")
    {
        const auto gain = perTenMillimetres(sweep.value(), [](const auto& s) { return s.camber * degrees; });

        REQUIRE(gain < 0.0);
        REQUIRE(gain == Catch::Approx(-0.177).margin(0.03));
    }

    SECTION("bump steer is very small, which is what a rear axle is for")
    {
        const auto steer = perTenMillimetres(sweep.value(), [](const auto& s) { return s.toe * degrees; });

        REQUIRE(std::abs(steer) < 0.05);

        // And across the whole travel the toe barely moves at all.
        auto worst = 0.0;
        for (const auto& sample : sweep->samples)
        {
            worst = std::max(worst, std::abs(sample.toe * degrees));
        }

        REQUIRE(worst < 0.45);
    }

    SECTION("and it scrubs half as much as the front")
    {
        const auto scrub = perTenMillimetres(sweep.value(), [](const auto& s) { return s.halfTrack * 1000.0; });

        REQUIRE(scrub == Catch::Approx(0.85).margin(0.10));
    }

    SECTION("the roll centre is where a rear axle's belongs")
    {
        auto atStatic = sweep->samples[sweep->samples.size() / 2];
        computeRollCentre(corner, atStatic);

        REQUIRE(atStatic.rollCentreHeight == Catch::Approx(0.065).margin(0.02));
    }

    SECTION("the travel is continuous and the solve never flips branch")
    {
        for (auto index = std::size_t{1}; index < sweep->samples.size(); index++)
        {
            REQUIRE(sweep->samples[index].wheelTravel > sweep->samples[index - 1].wheelTravel);
            REQUIRE(glm::distance(sweep->samples[index].upperBallJoint, sweep->samples[index - 1].upperBallJoint) <
                    0.01);
        }
    }
}

TEST_CASE("published data alone is not enough to build a corner from", "[physics][published]")
{
    // The limitation, asserted rather than described. Assetto Corsa's double-wishbone sections carry
    // no damper pickup points, because it states spring and damper rates *at the wheel*. Imported as
    // they stand, both damper mounts default to the same place and the motion ratio that comes out
    // is a plausible-looking number describing a linkage that does not exist — so the setup
    // validator refuses it, and a corner built from this data needs its damper geometry from
    // somewhere else.
    auto setup = CornerSetup{};
    setup.hardpoints = golfRear();

    const auto refused = validateCornerSetup(setup);
    REQUIRE_FALSE(refused.has_value());
    REQUIRE(refused.error().find("damper") != std::string::npos);
}
