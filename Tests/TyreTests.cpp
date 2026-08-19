#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine.physics;

using raceengine::evaluateTyre;
using raceengine::relaxTyre;
using raceengine::TyreForces;
using raceengine::tyreFriction;
using raceengine::TyreModel;
using raceengine::tyreSlip;
using raceengine::TyreSlip;
using raceengine::TyreState;

namespace
{

// The peak lateral force a tire makes at a given load, found by sweeping rather than by asking the
// model where its peak is — so the sweep is testing the curve rather than the bookkeeping.
double peakLateral(const TyreModel& model, const double load, const double grip = 1.0)
{
    auto best = 0.0;
    for (auto angle = 0.0; angle < 0.5; angle += 0.0005)
    {
        const auto forces = evaluateTyre(model, load, TyreSlip{.slipAngle = angle}, grip);
        best = std::max(best, std::abs(forces.lateral));
    }

    return best;
}

} // namespace

TEST_CASE("the lateral curve is a tire curve", "[physics][tyre]")
{
    const auto model = TyreModel{};
    const auto load = model.nominalLoad;

    SECTION("no slip, no force")
    {
        const auto forces = evaluateTyre(model, load, TyreSlip{}, 1.0);
        REQUIRE(forces.lateral == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(forces.longitudinal == Catch::Approx(0.0).margin(1e-12));
    }

    SECTION("it rises, peaks, and falls away")
    {
        auto previous = 0.0;
        auto peak = 0.0;
        auto peakAngle = 0.0;

        for (auto angle = 0.001; angle < 0.5; angle += 0.001)
        {
            const auto force = std::abs(evaluateTyre(model, load, TyreSlip{.slipAngle = angle}, 1.0).lateral);
            if (force > peak)
            {
                peak = force;
                peakAngle = angle;
            }
            previous = force;
        }

        // A peak somewhere in the range a road tire actually peaks in — six to twelve degrees.
        REQUIRE(peakAngle > 0.10);
        REQUIRE(peakAngle < 0.21);
        // And past it the force falls off rather than saturating flat, which is what makes the limit
        // findable and a slide recoverable.
        REQUIRE(previous < peak * 0.98);
    }

    SECTION("it is odd: steering the other way pulls the other way, exactly as hard")
    {
        const auto right = evaluateTyre(model, load, TyreSlip{.slipAngle = 0.08}, 1.0);
        const auto left = evaluateTyre(model, load, TyreSlip{.slipAngle = -0.08}, 1.0);

        REQUIRE(right.lateral == Catch::Approx(-left.lateral));
        REQUIRE(right.aligningMoment == Catch::Approx(-left.aligningMoment));
    }

    SECTION("a tire carrying nothing makes nothing")
    {
        REQUIRE(evaluateTyre(model, 0.0, TyreSlip{.slipAngle = 0.1}, 1.0).lateral == 0.0);
        REQUIRE(evaluateTyre(model, -50.0, TyreSlip{.slipAngle = 0.1}, 1.0).lateral == 0.0);
    }
}

TEST_CASE("friction falls with load, and non-linearly", "[physics][tyre][loadsensitivity]")
{
    // The mandatory property, and the one everything else in the car depends on: without it weight
    // transfer costs nothing, so no setup change can do anything and no amount of roll stiffness
    // will move the balance.
    const auto model = TyreModel{};

    const auto light = tyreFriction(model, model.lateralPeak, 2000.0, 1.0);
    const auto nominal = tyreFriction(model, model.lateralPeak, 4000.0, 1.0);
    const auto heavy = tyreFriction(model, model.lateralPeak, 8000.0, 1.0);

    REQUIRE(light > nominal);
    REQUIRE(nominal > heavy);
    REQUIRE(nominal == Catch::Approx(model.lateralPeak));

    SECTION("the fall is non-linear in load")
    {
        // Equal steps in load do not give equal steps in friction. A straight line would.
        const auto firstStep = tyreFriction(model, 1.0, 2000.0, 1.0) - tyreFriction(model, 1.0, 4000.0, 1.0);
        const auto secondStep = tyreFriction(model, 1.0, 4000.0, 1.0) - tyreFriction(model, 1.0, 6000.0, 1.0);

        REQUIRE(firstStep > secondStep * 1.2);
    }

    SECTION("and it never turns negative, however hard the car lands")
    {
        // The reason for a power law rather than the Magic Formula's usual linear term: linear load
        // sensitivity crosses zero somewhere above six times nominal load, and a tire with negative
        // friction shoves the car in whichever direction it was already sliding.
        REQUIRE(tyreFriction(model, model.lateralPeak, 100000.0, 1.0) > 0.0);
    }

    SECTION("an axle loses grip when its load is transferred across it")
    {
        // The consequence that matters, stated directly: two tires sharing 8000 N make more grip
        // evenly loaded than they do with it all on one side. This is why a stiffer bar loosens an
        // axle, and if this case fails no skidpad balance test below can mean anything.
        const auto even = peakLateral(model, 4000.0) * 2.0;
        const auto transferred = peakLateral(model, 6000.0) + peakLateral(model, 2000.0);

        REQUIRE(transferred < even);
    }

    SECTION("the surface multiplies grip without touching the coefficients")
    {
        // The seam the contact patch's blended grip arrives through, and the one a thermal model
        // will use later. Friction itself scales exactly; the swept peak only to the resolution of
        // the sweep, because halving the grip also halves the slip angle the peak sits at.
        REQUIRE(tyreFriction(model, model.lateralPeak, 4000.0, 0.5) ==
                Catch::Approx(tyreFriction(model, model.lateralPeak, 4000.0, 1.0) * 0.5));
        REQUIRE(peakLateral(model, 4000.0, 0.5) == Catch::Approx(peakLateral(model, 4000.0, 1.0) * 0.5).epsilon(1e-4));
    }
}

TEST_CASE("force builds over a distance, not over a time", "[physics][tyre][relaxation]")
{
    // Relaxation length is the other mandatory one. It is what makes the tire feel like a tire, and
    // it is also what removes the low-speed singularity — so it has to behave the same at every
    // speed, which is exactly what "over a distance" means.
    const auto model = TyreModel{};

    const auto riseDistance = [&model](const double speed)
    {
        auto state = TyreState{};
        const auto lateralSlipVelocity = speed * std::tan(0.05);

        // Steady state, for comparison.
        const auto settled = model.lateralRelaxation * lateralSlipVelocity / speed;

        // One time constant is 1 - 1/e of the way there. Interpolated within the step it is
        // crossed in rather than rounded up to it: at 10 m/s a tick is 28 mm of road, which is 5%
        // of the distance being measured and would swamp the property being tested.
        const auto target = settled * (1.0 - 1.0 / 2.718281828459045);

        auto travelled = 0.0;
        auto previous = 0.0;

        for (auto tick = 0; tick < 200000; tick++)
        {
            relaxTyre(model, state, speed, 0.0, lateralSlipVelocity, 1.0 / 360.0);
            const auto step = speed / 360.0;

            if (state.lateralDeflection >= target)
            {
                const auto fraction = (target - previous) / (state.lateralDeflection - previous);
                return travelled + fraction * step;
            }

            travelled += step;
            previous = state.lateralDeflection;
        }

        return travelled;
    };

    const auto slow = riseDistance(10.0);
    const auto fast = riseDistance(50.0);

    // One relaxation length to reach 63% of the way there, whatever the speed.
    REQUIRE(slow == Catch::Approx(model.lateralRelaxation).epsilon(0.05));
    REQUIRE(fast == Catch::Approx(model.lateralRelaxation).epsilon(0.05));
    REQUIRE(fast == Catch::Approx(slow).epsilon(0.05));
}

TEST_CASE("the slip the tire works at recovers the textbook one at steady state", "[physics][tyre][relaxation]")
{
    const auto model = TyreModel{};

    auto state = TyreState{};
    constexpr auto speed = 25.0;
    constexpr auto trueAngle = 0.06;

    for (auto tick = 0; tick < 3600; tick++)
    {
        relaxTyre(model, state, speed, 0.0, speed * std::tan(trueAngle), 1.0 / 360.0);
    }

    REQUIRE(tyreSlip(model, state).slipAngle == Catch::Approx(trueAngle).epsilon(1e-3));

    SECTION("and the slip ratio likewise")
    {
        auto rolling = TyreState{};
        // Ten percent slip: the patch slides backwards at a tenth of road speed.
        for (auto tick = 0; tick < 3600; tick++)
        {
            relaxTyre(model, rolling, speed, -0.1 * speed, 0.0, 1.0 / 360.0);
        }

        REQUIRE(tyreSlip(model, rolling).slipRatio == Catch::Approx(-0.1).epsilon(1e-3));
    }
}

TEST_CASE("a stopped tire is a spring, not a division by zero", "[physics][tyre][lowspeed]")
{
    // Criterion 8's foundation. Nothing here divides by wheel speed: what is integrated is the
    // carcass deflection, which at a standstill simply accumulates, so a stationary car on a slope
    // is held by a tire behaving like a tire rather than by a special case.
    const auto model = TyreModel{};

    auto state = TyreState{};
    for (auto tick = 0; tick < 360; tick++)
    {
        // Being dragged sideways at walking pace with the wheel not turning at all.
        relaxTyre(model, state, 0.0, 0.0, 0.001, 1.0 / 360.0);
    }

    const auto slip = tyreSlip(model, state);
    const auto forces = evaluateTyre(model, 4000.0, slip, 1.0);

    REQUIRE(std::isfinite(slip.slipAngle));
    REQUIRE(std::isfinite(forces.lateral));
    // And it resists: the force opposes the direction it is being dragged.
    REQUIRE(forces.lateral < 0.0);

    SECTION("creeping at walking pace is finite and smooth")
    {
        auto creeping = TyreState{};
        auto worstJump = 0.0;
        auto previous = 0.0;

        for (auto tick = 0; tick < 3600; tick++)
        {
            const auto speed = 0.3;
            relaxTyre(model, creeping, speed, -0.02 * speed, 0.01, 1.0 / 360.0);

            const auto force = evaluateTyre(model, 4000.0, tyreSlip(model, creeping), 1.0).lateral;
            REQUIRE(std::isfinite(force));
            worstJump = std::max(worstJump, std::abs(force - previous));
            previous = force;
        }

        // No tick-to-tick spikes, which is the failure a slip-ratio clamp produces at low speed.
        REQUIRE(worstJump < 200.0);
    }
}

TEST_CASE("braking and cornering are not both available at once", "[physics][tyre][combined]")
{
    // The friction ellipse. Full braking and full cornering must not be simultaneously on offer.
    const auto model = TyreModel{};
    const auto load = model.nominalLoad;

    const auto pureLateral = std::abs(evaluateTyre(model, load, TyreSlip{.slipAngle = 0.14}, 1.0).lateral);
    const auto pureLongitudinal = std::abs(evaluateTyre(model, load, TyreSlip{.slipRatio = 0.12}, 1.0).longitudinal);

    REQUIRE(pureLateral > 0.0);
    REQUIRE(pureLongitudinal > 0.0);

    const auto combined = evaluateTyre(model, load, TyreSlip{.slipRatio = 0.12, .slipAngle = 0.14}, 1.0);

    SECTION("asking for both gives less of each")
    {
        REQUIRE(std::abs(combined.lateral) < pureLateral);
        REQUIRE(std::abs(combined.longitudinal) < pureLongitudinal);
    }

    SECTION("and the total stays inside the circle either alone could reach")
    {
        const auto total = std::hypot(combined.lateral, combined.longitudinal);
        REQUIRE(total < std::max(pureLateral, pureLongitudinal) * 1.05);
    }

    SECTION("braking hard leaves very little to steer with")
    {
        const auto locked = evaluateTyre(model, load, TyreSlip{.slipRatio = 0.9, .slipAngle = 0.14}, 1.0);
        REQUIRE(std::abs(locked.lateral) < pureLateral * 0.35);
    }
}

TEST_CASE("the aligning moment goes light before the grip does", "[physics][tyre]")
{
    // Pneumatic trail collapsing ahead of the lateral peak is the single most useful thing a tire
    // tells a driver, and the reason a real wheel goes light in the hands at the limit.
    const auto model = TyreModel{};
    const auto load = model.nominalLoad;

    auto peakMomentAngle = 0.0;
    auto peakMoment = 0.0;
    auto peakForceAngle = 0.0;
    auto peakForce = 0.0;

    for (auto angle = 0.001; angle < 0.4; angle += 0.001)
    {
        const auto forces = evaluateTyre(model, load, TyreSlip{.slipAngle = angle}, 1.0);

        if (std::abs(forces.aligningMoment) > peakMoment)
        {
            peakMoment = std::abs(forces.aligningMoment);
            peakMomentAngle = angle;
        }

        if (std::abs(forces.lateral) > peakForce)
        {
            peakForce = std::abs(forces.lateral);
            peakForceAngle = angle;
        }
    }

    REQUIRE(peakMoment > 0.0);
    REQUIRE(peakMomentAngle < peakForceAngle);

    // And it self-aligns: the moment acts to *reduce* the slip angle, whichever way the wheel is
    // slipping. A sign error here gives a tire that steers itself further into the corner, which
    // reads as a terminal instability rather than as a sign error.
    for (const auto angle : {-0.12, -0.05, 0.05, 0.12})
    {
        const auto steered = evaluateTyre(model, load, TyreSlip{.slipAngle = angle}, 1.0);
        REQUIRE(steered.aligningMoment * angle < 0.0);
    }
}

TEST_CASE("the seams the deferred models need are already there", "[physics][tyre]")
{
    auto model = TyreModel{};
    const auto load = model.nominalLoad;

    SECTION("grip is a runtime scale, not something baked into the coefficients")
    {
        model.gripScale = 0.8;
        REQUIRE(peakLateral(model, load) == Catch::Approx(peakLateral(TyreModel{}, load) * 0.8).epsilon(1e-6));
    }

    SECTION("the peak can be moved in slip without moving its height")
    {
        model.peakSlipScale = 1.4;

        auto shifted = 0.0;
        auto shiftedAngle = 0.0;
        for (auto angle = 0.001; angle < 0.6; angle += 0.0005)
        {
            const auto force = std::abs(evaluateTyre(model, load, TyreSlip{.slipAngle = angle}, 1.0).lateral);
            if (force > shifted)
            {
                shifted = force;
                shiftedAngle = angle;
            }
        }

        REQUIRE(shifted == Catch::Approx(peakLateral(TyreModel{}, load)).epsilon(0.02));
        REQUIRE(shiftedAngle > 0.15);
    }

    SECTION("slip power is computed even though nothing consumes it")
    {
        const auto forces = evaluateTyre(model, load, TyreSlip{.slipRatio = 0.1, .slipAngle = 0.1}, 1.0, -2.0, 1.5);

        REQUIRE(forces.slipPower > 0.0);
        REQUIRE(std::isfinite(forces.slipPower));
    }

    SECTION("grip used says how near the limit the tire is")
    {
        const auto gentle = evaluateTyre(model, load, TyreSlip{.slipAngle = 0.02}, 1.0);
        const auto hard = evaluateTyre(model, load, TyreSlip{.slipAngle = 0.20}, 1.0);

        REQUIRE(gentle.gripUsed < hard.gripUsed);
        REQUIRE(gentle.gripUsed > 0.0);
    }
}
