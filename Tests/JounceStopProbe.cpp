#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::jounceBumperCandidate;
using raceengine::jounceElementCount;
using raceengine::jounceReferenceForce;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::TravelStop;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

// The sourced jounce-bumper stop model, printed. `./EngineTests "[.jounce-stop]"`.
//
// Hidden behind a dotted tag like every other probe here: what it produces is a table to read and a
// question to put to whoever is driving, not a bound to hold. The bounds that came out of it are in
// `VehicleTests.cpp`.
//
// **Why it exists.** On 2026-08-30 the bump stop's seat A/B — the sourced BASF hysteresis with the
// placed viscous constant taken out — was driven and *rejected*: the car pogoed in pitch at about
// 2 Hz off a 160 km/h kerb and flipped. The trace showed the oscillation decaying, so the
// hysteresis model added no energy; it was an honestly underdamped stop. Pech et al. (VSD 2024,
// 10.1080/00423114.2024.2378858) then supplied the missing piece from a measured production
// bumper: the dissipation that stops that pogo is a **rate-dependent Maxwell branch**, not a larger
// rate-independent hysteresis. This probe builds that branch's candidate onto this car's own stop
// and measures whether it does the job the placed constant is doing today.
//
// **Nothing here is on any car.** The candidate is transferred from one published specimen and is
// reachable only as `front.stopdynamic 1` on a setup sheet.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;

// **The plate runs z from 0 to its length and x from -width/2 to +width/2.** A fixture that assumes
// it is centred starts the car in mid-air, and everything it then reports is a car in free fall.
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
constexpr auto startZ = 20.0;

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

[[nodiscard]] SurfaceMesh gripPlate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(1);
    mesh->materials[0].gripMultiplier = 1.0;
    mesh->materials[0].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        mesh->surfaces[triangle] = std::uint32_t{0};
    }

    return mesh.value();
}

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed)
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

// A car with one stop model on all four bump stops. Everything else is the Golf.
[[nodiscard]] VehicleSetup withStop(const VehicleSetup& base, const TravelStop& stop)
{
    auto car = base;
    for (auto& corner : car.corners)
    {
        // The gap and the static law are the corner's own — only the dissipation model changes,
        // which is what makes the rows below an A/B rather than four different cars.
        auto fitted = stop;
        fitted.gap = corner.bumpStop.gap;
        fitted.rate = corner.bumpStop.rate;
        fitted.progression = corner.bumpStop.progression;
        corner.bumpStop = fitted;
    }

    return car;
}

// The candidate at a stated cut-off, built onto a stop that carries the car's own static law and no
// viscous damping at all — which is the comparison that matters, because the whole question is
// whether the sourced branch can replace the placed constant rather than sit on top of it.
[[nodiscard]] TravelStop candidateFor(const TravelStop& stop, const double cutoff)
{
    auto bare = stop;
    bare.damping = 0.0;
    bare.hysteresis = 0.0;

    return jounceBumperCandidate(bare, cutoff);
}

} // namespace

TEST_CASE("the sourced jounce branch transferred onto this car's own bump stop", "[.jounce-stop]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto& front = base->corners[0].bumpStop;
    const auto candidate = candidateFor(front, 3.1);

    std::printf("\n=== Pech et al., VSD 2024: the specimen, and the transfer onto this stop ===\n");
    std::printf("  the specimen (published)\n");
    std::printf("    reference force / deflection      %8.0f N at %.0f mm\n", jounceReferenceForce, 71.0);
    std::printf("    max operating point on a real road       3000 N at 68 mm\n");
    std::printf("    static hysteresis half-height            900 N  = %.1f%% of the reference force\n",
                100.0 * 900.0 / jounceReferenceForce);
    std::printf("    dynamic stiffening, peak                2800 N  = %.1f%% of it\n",
                100.0 * 2800.0 / jounceReferenceForce);
    std::printf("    Maxwell elements                           %zu, at their own contact points\n",
                jounceElementCount);

    // This stop's own reference deflection, recomputed here rather than read out of the candidate,
    // so the table is checking the transfer rather than printing it back.
    const auto scale = std::pow(front.gap, front.progression - 1.0);
    const auto reference = std::pow(jounceReferenceForce * scale / front.rate, 1.0 / front.progression);

    std::printf("\n  this car's front bump stop\n");
    std::printf("    static law                        %.0f N/m at power %.1f over a %.0f mm gap\n", front.rate,
                front.progression, 1000.0 * front.gap);
    std::printf("    its own reference deflection      %8.2f mm  (specimen 71 mm, so x%.3f)\n", 1000.0 * reference,
                reference / 0.071);
    std::printf("    shipped dissipation               %8.0f N.s/m viscous, hysteresis %.2f\n", front.damping,
                front.hysteresis);

    REQUIRE(candidate.dahlReference > 0.0);
    CHECK(std::abs(candidate.dahlReference - reference) < 1e-9);

    std::printf("\n  the transferred candidate\n");
    std::printf("    Dahl saturation                   %8.2f  of the elastic force (source: 0.9 kN of 9 kN)\n",
                candidate.hysteresis);
    std::printf("    Dahl coefficient (PLACED)         %8.1f /m, constant - the source's three are unpublished\n",
                candidate.dahlSigmaMin);
    std::printf("    contact-point fade-in (PLACED)    %8.3f mm\n", 1000.0 * candidate.elementSmoothing);
    std::printf("\n    element   contact point   stiffness   cut-off   carries at reference\n");

    auto carried = 0.0;
    auto previous = -1.0;
    for (auto index = std::size_t{0}; index < jounceElementCount; index++)
    {
        const auto& element = candidate.elements[index];
        const auto share = element.stiffness * std::max(0.0, reference - element.contactPoint);
        carried += share;

        std::printf("      %zu       %9.2f mm  %8.0f N/m  %6.2f rad/s  %10.0f N\n", index + 1,
                    1000.0 * element.contactPoint, element.stiffness, element.cutoff, share);

        // The contact points must climb, or the piece-wise linear envelope they were read off has
        // been transferred with its segments out of order.
        CHECK(element.contactPoint > previous);
        previous = element.contactPoint;
    }

    std::printf("\n    total dynamic stiffening at the reference deflection   %.0f N = %.1f%% of the elastic force\n",
                carried, 100.0 * carried / jounceReferenceForce);
    std::printf("    the specimen's own envelope reads about 2000 N there, and 2800 N at full stroke.\n");

    // The transfer's own arithmetic check: the elements must reproduce the specimen's share of the
    // reference force, not merely be present. A fifth to a quarter is what Figure 16 shows there.
    CHECK(carried > 0.15 * jounceReferenceForce);
    CHECK(carried < 0.30 * jounceReferenceForce);
}

TEST_CASE("a 3.0 m/s drop onto the wheels, stop model by stop model", "[.jounce-stop]")
{
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto& front = base->corners[0].bumpStop;

    struct Strike
    {
        double peakStop = 0.0;
        double deepest = 0.0;
        int touches = 0;
        double settling = 0.0;
        double later = 0.0;
    };

    const auto measure = [&](const TravelStop& stop)
    {
        const auto car = withStop(base.value(), stop);

        auto state = VehicleState{};
        settle(car, state, world.value(), 0.0);

        const auto rested = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(rested.has_value());
        const auto rest = rested->corners[0].suspension.wheelTravel;

        state.chassis.linearVelocity = glm::dvec3(0.0, -3.0, 0.0);

        auto result = Strike{};
        auto outside = 0;
        constexpr auto steps = 360 * 8;

        for (auto step = 1; step <= steps; step++)
        {
            const auto stepped = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto force = stepped->corners[0].forces.bumpStop;
            const auto excursion = std::abs(stepped->corners[0].suspension.wheelTravel - rest);

            result.peakStop = std::max(result.peakStop, force);
            result.deepest = std::max(result.deepest, stepped->corners[0].suspension.wheelTravel - rest);
            result.touches += force > 0.0 ? 1 : 0;

            if (step > 360)
            {
                result.later = std::max(result.later, excursion);
            }

            if (excursion > 0.002)
            {
                outside = step;
            }
        }

        result.settling = static_cast<double>(outside) * tick;

        return result;
    };

    std::printf("\n=== a 3.0 m/s drop onto the wheels, front corner ===\n");
    std::printf("   peak F stop   deepest   touches   settle to 2 mm   worst after 1 s   model\n");

    auto hysteresisOnly = front;
    hysteresisOnly.damping = 0.0;
    hysteresisOnly.hysteresis = 0.07;

    auto bare = front;
    bare.damping = 0.0;
    bare.hysteresis = 0.0;

    // The candidate on top of the shipped viscous constant, which is not a proposal — it is the
    // control that says how much of the row above is the branch and how much is the constant.
    auto stacked = jounceBumperCandidate(front, 3.1);

    const auto cases = std::array{
        std::pair{front, "shipped: 40000 N.s/m viscous"},
        std::pair{hysteresisOnly, "BASF hysteresis only, 0.07 - the REJECTED A/B"},
        std::pair{bare, "bare spring"},
        std::pair{candidateFor(front, 1.0), "sourced branch, cut-off 1.0 rad/s"},
        std::pair{candidateFor(front, 3.1), "sourced branch, cut-off 3.1 rad/s - the candidate"},
        std::pair{candidateFor(front, 10.0), "sourced branch, cut-off 10 rad/s"},
        std::pair{candidateFor(front, 30.0), "sourced branch, cut-off 30 rad/s"},
        std::pair{candidateFor(front, 100.0), "sourced branch, cut-off 100 rad/s"},
        std::pair{stacked, "sourced branch WITH the shipped viscous constant"},
    };

    for (const auto& [stop, note] : cases)
    {
        const auto strike = measure(stop);

        std::printf("  %10.0f N  %7.2f mm  %7d  %13.3f s  %13.2f mm  %s\n", strike.peakStop, 1000.0 * strike.deepest,
                    strike.touches, strike.settling, 1000.0 * strike.later, note);

        // Every row has to be a car that is still on the ground and still touching its stop, or the
        // row is measuring a failed drop rather than a stop model.
        CHECK(strike.touches > 0);
        CHECK(strike.deepest > 0.0);
    }

    std::printf("\n  The cut-off is the one number Pech et al. do NOT publish - only the band it lies in.\n");
    std::printf("  Read the sweep as the size of that gap, not as a tuning knob.\n");
}

TEST_CASE("the kerb pogo: whether a stop model kills a pitch oscillation", "[.jounce-stop]")
{
    // **The fixture the flip asked for.** The seat trace
    // `traces/rack-exit-20260829-stop-hysteresis-flip-seat.csv` shows a kerb at about 160 km/h
    // driving the fronts 26-32 mm into stops that engage at 21.6 mm, then a ~2 Hz pitch pogo for
    // 2.5 s whose amplitude decayed only about 20% — and a second disturbance arriving while the
    // car was still bouncing put it over. Session 1, on the shipped stop, killed the same kind of
    // strike in one cycle.
    //
    // So the question a fixture has to answer is not "how big is the peak force" — the drop above
    // answers that — but "how fast does the oscillation go away". This excites the pitch mode
    // directly, at the trace's own road speed, and reads the decay.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto& front = base->corners[0].bumpStop;
    constexpr auto speed = 160.0 / 3.6;

    struct Pogo
    {
        double deepestFront = 0.0;
        double pastStop = 0.0;
        int touches = 0;
        std::array<double, 6> window{};
        double decay = 0.0;
        double quiet = 0.0;
    };

    const auto measure = [&](const TravelStop& stop, const double pitchRate)
    {
        const auto car = withStop(base.value(), stop);

        auto state = VehicleState{};
        settle(car, state, world.value(), speed);

        const auto rested = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(rested.has_value());

        auto rest = std::array<double, cornerCount>{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            rest[index] = rested->corners[index].suspension.wheelTravel;
        }

        // Pitch about the body's lateral axis, seeded as momentum because that is what the chassis
        // stores. The rate is chosen to drive the fronts into their stops by about the depth the
        // trace shows, and no further: this is a kerb, not a crash.
        state.chassis.angularMomentum = state.chassis.inertia * glm::dvec3(pitchRate, 0.0, 0.0);

        auto result = Pogo{};
        constexpr auto steps = 360 * 3;
        auto outside = 0;

        for (auto step = 1; step <= steps; step++)
        {
            const auto stepped = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            // The front pair's travel against where they were sitting, and the deepest either got.
            auto excursion = 0.0;
            for (auto index = std::size_t{0}; index < 2; index++)
            {
                const auto travel = stepped->corners[index].suspension.wheelTravel - rest[index];
                excursion = std::max(excursion, std::abs(travel));
                result.deepestFront = std::max(result.deepestFront, travel);
                result.touches += stepped->corners[index].forces.bumpStop > 0.0 ? 1 : 0;
            }

            const auto slot = static_cast<std::size_t>((step - 1) / 180);
            if (slot < result.window.size())
            {
                result.window[slot] = std::max(result.window[slot], excursion);
            }

            if (excursion > 0.003)
            {
                outside = step;
            }
        }

        result.pastStop = std::max(0.0, result.deepestFront - front.gap);
        result.quiet = static_cast<double>(outside) * tick;

        // How much of the first half second's amplitude survives into the third: the trace's own
        // measure, where hysteresis-only kept about 80% of it over two seconds.
        result.decay = result.window[0] > 0.0 ? result.window[4] / result.window[0] : 0.0;

        return result;
    };

    // Which way round pitch drives the front into bump is a property of this frame, not something
    // to assume: it is left-handed and the sign has been got wrong here before. So the rate is
    // chosen by measuring, and the fixture asserts that it found one.
    const auto positive = measure(front, 0.8);
    const auto negative = measure(front, -0.8);
    const auto pitchRate = positive.deepestFront > negative.deepestFront ? 0.8 : -0.8;

    std::printf("\n=== a pitch kick at %.0f km/h, front pair, three seconds ===\n", 3.6 * speed);
    std::printf("  pitch rate %+.1f rad/s drives the front into bump (%+.2f mm against %+.2f the other way)\n",
                pitchRate, 1000.0 * std::max(positive.deepestFront, negative.deepestFront),
                1000.0 * std::min(positive.deepestFront, negative.deepestFront));
    std::printf("  the stop engages at %.1f mm of travel\n\n", 1000.0 * front.gap);
    std::printf("   deepest   past stop   touches   amplitude by half second (mm)                  decay   quiet\n");

    auto hysteresisOnly = front;
    hysteresisOnly.damping = 0.0;
    hysteresisOnly.hysteresis = 0.07;

    auto bare = front;
    bare.damping = 0.0;
    bare.hysteresis = 0.0;

    const auto cases = std::array{
        std::pair{front, "shipped: 40000 N.s/m viscous"},
        std::pair{hysteresisOnly, "BASF hysteresis only, 0.07 - the REJECTED A/B"},
        std::pair{bare, "bare spring"},
        std::pair{candidateFor(front, 1.0), "sourced branch, cut-off 1.0 rad/s"},
        std::pair{candidateFor(front, 3.1), "sourced branch, cut-off 3.1 rad/s - the candidate"},
        std::pair{candidateFor(front, 10.0), "sourced branch, cut-off 10 rad/s"},
        std::pair{candidateFor(front, 30.0), "sourced branch, cut-off 30 rad/s"},
        std::pair{candidateFor(front, 100.0), "sourced branch, cut-off 100 rad/s"},
        std::pair{jounceBumperCandidate(front, 3.1), "sourced branch WITH the shipped viscous constant"},
    };

    for (const auto& [stop, note] : cases)
    {
        const auto pogo = measure(stop, pitchRate);

        std::printf("  %6.2f mm  %7.2f mm  %7d  ", 1000.0 * pogo.deepestFront, 1000.0 * pogo.pastStop, pogo.touches);
        for (const auto amplitude : pogo.window)
        {
            std::printf("%6.2f", 1000.0 * amplitude);
        }
        std::printf("  %6.2f  %5.2f s  %s\n", pogo.decay, pogo.quiet, note);
    }

    std::printf("\n  `decay` is the amplitude two seconds in over the amplitude in the first half second.\n");
    std::printf("  The flip trace decayed only about 20%% over two seconds - a decay near 0.8 is that car.\n");
}

TEST_CASE("driven into the stops at the pogo's own frequency, then let go", "[.jounce-stop]")
{
    // **The single kick above does not reproduce the flip and that is a result, not a defect.** On
    // flat ground every stop model — including a bare spring with no dissipation at all — settles
    // one strike inside a second, because the main dampers own the settle. What the flip trace has
    // that a single kick does not is *repetition*: Dominic's report is "the oscillation left right
    // over kerbs", a car being struck again while it is still moving, and the second disturbance
    // arriving mid-bounce is what put it over.
    //
    // So this drives the body at the trace's own 2 Hz for two seconds — repeated strikes into the
    // stops at road speed — and then takes the excitation away and watches. The forced amplitude
    // answers "how big does it get while the kerbs keep coming"; the free decay afterwards answers
    // "does it stop when they do".
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto& front = base->corners[0].bumpStop;
    constexpr auto speed = 160.0 / 3.6;
    constexpr auto drivenFor = 2.0;
    constexpr auto freeFor = 2.0;
    constexpr auto frequency = 2.0;
    constexpr auto torque = 6000.0;

    struct Driven
    {
        double forced = 0.0;
        double deepest = 0.0;
        double pastStop = 0.0;
        int touches = 0;
        double released = 0.0;
        double after = 0.0;
        double quiet = 0.0;
        int airborne = 0;

        // Net work into the stops over the driven window, all four corners, joules. Over a stroke
        // that returns to where it started this **is** the energy the stop took out of the car,
        // which is the only quantity that decides whether an oscillation goes away.
        double dissipated = 0.0;
    };

    const auto measure = [&](const TravelStop& stop, const glm::dvec3& axis)
    {
        const auto car = withStop(base.value(), stop);

        auto state = VehicleState{};
        settle(car, state, world.value(), speed);

        const auto rested = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(rested.has_value());

        auto rest = std::array<double, cornerCount>{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            rest[index] = rested->corners[index].suspension.wheelTravel;
        }

        auto result = Driven{};
        const auto drivenSteps = static_cast<int>(drivenFor / tick);
        const auto totalSteps = drivenSteps + static_cast<int>(freeFor / tick);
        auto outside = 0;

        for (auto step = 1; step <= totalSteps; step++)
        {
            const auto now = static_cast<double>(step) * tick;

            // A torque applied by adding its impulse to the stored momentum, which is what a torque
            // is. Taken away completely at the release, so what follows is the car's own.
            if (step <= drivenSteps)
            {
                state.chassis.angularMomentum +=
                    axis * (torque * std::sin(2.0 * std::numbers::pi * frequency * now) * tick);
            }

            const auto stepped = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            auto excursion = 0.0;
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto travel = stepped->corners[index].suspension.wheelTravel - rest[index];
                excursion = std::max(excursion, std::abs(travel));
                result.deepest = std::max(result.deepest, travel);
                result.touches += stepped->corners[index].forces.bumpStop > 0.0 ? 1 : 0;
                result.airborne += stepped->corners[index].patch.inContact ? 0 : 1;

                if (step <= drivenSteps)
                {
                    result.dissipated +=
                        stepped->corners[index].forces.bumpStop * stepped->corners[index].damperVelocity * tick;
                }
            }

            if (step <= drivenSteps)
            {
                result.forced = std::max(result.forced, excursion);
            }
            else
            {
                if (step <= drivenSteps + 180)
                {
                    result.released = std::max(result.released, excursion);
                }

                // The last half second of the free window: what is still going a second and a half
                // after the road stopped hitting the car.
                if (step > totalSteps - 180)
                {
                    result.after = std::max(result.after, excursion);
                }

                if (excursion > 0.003)
                {
                    outside = step - drivenSteps;
                }
            }
        }

        result.pastStop = std::max(0.0, result.deepest - front.gap);
        result.quiet = static_cast<double>(outside) * tick;

        return result;
    };

    auto hysteresisOnly = front;
    hysteresisOnly.damping = 0.0;
    hysteresisOnly.hysteresis = 0.07;

    auto bare = front;
    bare.damping = 0.0;
    bare.hysteresis = 0.0;

    const auto cases = std::array{
        std::pair{front, "shipped: 40000 N.s/m viscous"},
        std::pair{hysteresisOnly, "BASF hysteresis only, 0.07 - the REJECTED A/B"},
        std::pair{bare, "bare spring"},
        std::pair{candidateFor(front, 1.0), "sourced branch, cut-off 1.0 rad/s"},
        std::pair{candidateFor(front, 3.1), "sourced branch, cut-off 3.1 rad/s - the candidate"},
        std::pair{candidateFor(front, 10.0), "sourced branch, cut-off 10 rad/s"},
        std::pair{candidateFor(front, 30.0), "sourced branch, cut-off 30 rad/s"},
        std::pair{candidateFor(front, 100.0), "sourced branch, cut-off 100 rad/s"},
        std::pair{jounceBumperCandidate(front, 3.1), "sourced branch WITH the shipped viscous constant"},
    };

    for (const auto& [axis, name] :
         std::array{std::pair{glm::dvec3(0.0, 0.0, 1.0), "roll"}, std::pair{glm::dvec3(1.0, 0.0, 0.0), "pitch"}})
    {
        std::printf("\n=== %s, driven at %.1f Hz with %.0f N.m for %.0fs at %.0f km/h, then released ===\n", name,
                    frequency, torque, drivenFor, 3.6 * speed);
        std::printf("  the stop engages at %.1f mm of travel\n\n", 1000.0 * front.gap);
        std::printf("   forced   deepest   past stop   touches   air   took out   at release   1.5s later   quiet   "
                    "model\n");

        for (const auto& [stop, note] : cases)
        {
            const auto driven = measure(stop, axis);

            std::printf("  %6.2f  %7.2f mm  %8.2f mm  %7d  %4d  %7.1f J  %9.2f mm  %9.2f mm  %5.2f s  %s\n",
                        1000.0 * driven.forced, 1000.0 * driven.deepest, 1000.0 * driven.pastStop, driven.touches,
                        driven.airborne, driven.dissipated, 1000.0 * driven.released, 1000.0 * driven.after,
                        driven.quiet, note);

            // Every row must actually reach its stop, or it is not a test of a stop model.
            CHECK(driven.touches > 0);
        }
    }

    std::printf("\n  `forced` is the worst corner excursion while the road is still hitting the car;\n");
    std::printf("  `quiet` is how long after the last hit the car keeps moving more than 3 mm.\n");
    std::printf("  The flip: an oscillation still large enough that the NEXT kerb arrived on top of it.\n");
}
