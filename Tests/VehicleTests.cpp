#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::Corner;
using raceengine::cornerCount;
using raceengine::Curve;
using raceengine::generateProvingGround;
using raceengine::linearDamper;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TravelStop;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::VehicleStep;

namespace
{

constexpr auto vehicleStep = 1.0 / 360.0;
constexpr auto gravity = 9.80665;

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

// A big flat plate, which is what settling, the skidpad and the coastdown all want. The featured
// proving ground is for the cases that need a kerb.
ProvingGroundDescriptor flatPlate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 120.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 1.0;
    descriptor.features = {};

    return descriptor;
}

} // namespace

TEST_CASE("a damper is a curve, not a number", "[physics][vehicle]")
{
    // The interface the brief asks for: low and high speed knees are simply more points, and a
    // linear damper is the special case rather than the model.
    const auto damper = linearDamper(4000.0, 8000.0);

    REQUIRE(damper.at(0.0) == Catch::Approx(0.0));
    REQUIRE(damper.at(0.1) == Catch::Approx(400.0));
    // Rebound is not bump, which is true of every real damper ever built.
    REQUIRE(damper.at(-0.1) == Catch::Approx(-800.0));

    SECTION("a knee is two more points and nothing else changes")
    {
        const auto kneed = Curve{.points = {glm::dvec2(-1.0, -6000.0), glm::dvec2(-0.05, -400.0), glm::dvec2(0.0, 0.0),
                                            glm::dvec2(0.05, 250.0), glm::dvec2(1.0, 3000.0)}};

        REQUIRE(kneed.at(0.025) == Catch::Approx(125.0));
        // Past the knee the slope is the high-speed one, which is softer here.
        REQUIRE(kneed.at(0.5) == Catch::Approx(250.0 + (3000.0 - 250.0) * (0.5 - 0.05) / 0.95));
    }

    SECTION("past the last point the end value is held rather than extrapolated")
    {
        REQUIRE(damper.at(500.0) == Catch::Approx(damper.at(5.0)));
        REQUIRE(damper.at(-500.0) == Catch::Approx(damper.at(-5.0)));
    }
}

TEST_CASE("a travel stop comes in softly and then not at all", "[physics][vehicle]")
{
    const auto stop = TravelStop{.gap = 0.04, .rate = 300000.0, .progression = 2.5};

    REQUIRE(stop.force(-0.01) == 0.0);
    REQUIRE(stop.force(0.0) == 0.0);

    // Progressive: twice the deflection is more than twice the force, which is what stops a bump
    // stop either being transparent or hammering the chassis the instant it touches.
    const auto light = stop.force(0.005);
    const auto heavy = stop.force(0.010);

    REQUIRE(light > 0.0);
    REQUIRE(heavy > 2.0 * light);
}

TEST_CASE("the placeholder car is a car", "[physics][vehicle]")
{
    const auto setup = placeholderSedan();
    REQUIRE(setup.has_value());

    // Every corner's geometry has already been swept and validated inside the factory; this is the
    // mass ledger it was built around.
    const auto properties = raceengine::computeMassProperties(setup->sprung);
    REQUIRE(properties.has_value());

    const auto total = properties->mass + setup->unsprungMass();
    REQUIRE(total == Catch::Approx(1352.0));
    REQUIRE(setup->unsprungMass() == Catch::Approx(152.0));

    // Spring free lengths were solved for, not guessed: each is longer than the damper at design,
    // so every corner is preloaded rather than slack.
    for (const auto& corner : setup->corners)
    {
        REQUIRE(corner.springFreeLength > 0.4);
        REQUIRE(corner.springRate > 0.0);
    }
}

TEST_CASE("the car settles on its springs", "[physics][vehicle][settle]")
{
    // Acceptance criterion 4, end to end: dropped, it must settle to a stable ride height with no
    // residual oscillation and no jitter, and the corner loads must sum to its weight and match the
    // distribution it was built with.
    const JoltGuard jolt;

    const auto setup = placeholderSedan();
    REQUIRE(setup.has_value());

    const auto mesh = generateProvingGround(flatPlate());
    REQUIRE(mesh.has_value());

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    // Half a metre up. The design ride height puts the contact patches on y = 0 with the centre of
    // gravity 0.52 m above them, so this is a 0.5 m drop.
    state.chassis.position = glm::dvec3(0.0, 1.02, 20.0);

    const auto input = VehicleInput{};

    auto heights = std::vector<double>{};
    auto last = VehicleStep{};

    // Six seconds, which is many times the ride frequency's period.
    for (auto tick = 0; tick < 2160; tick++)
    {
        auto stepped = stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), vehicleStep);
        REQUIRE(stepped.has_value());

        last = stepped.value();
        heights.push_back(state.chassis.position.y);
    }

    SECTION("it comes to rest")
    {
        REQUIRE(std::abs(state.chassis.linearVelocity.y) < 1e-3);

        // Over the last half second the ride height must be still — not merely bounded, still. A
        // millimetre of residual motion here is the jitter the criterion rules out.
        const auto tail = std::vector<double>(heights.end() - 180, heights.end());
        const auto lowest = *std::min_element(tail.begin(), tail.end());
        const auto highest = *std::max_element(tail.begin(), tail.end());

        REQUIRE(highest - lowest < 1e-4);
    }

    SECTION("it settles somewhere sensible rather than on its stops")
    {
        // Somewhere near the design height, having sunk into its tires and springs a little.
        REQUIRE(state.chassis.position.y > 0.40);
        REQUIRE(state.chassis.position.y < 0.53);

        for (const auto& corner : last.corners)
        {
            REQUIRE(corner.patch.inContact);
            // Neither stop is carrying the car; the springs are.
            REQUIRE(corner.forces.bumpStop == Catch::Approx(0.0).margin(1.0));
            REQUIRE(corner.forces.droopStop == Catch::Approx(0.0).margin(1.0));
        }
    }

    SECTION("the corner loads sum to the car's weight")
    {
        auto total = 0.0;
        for (const auto& corner : last.corners)
        {
            total += corner.forces.tireVertical;
        }

        const auto weight = 1352.0 * gravity;
        REQUIRE(total == Catch::Approx(weight).epsilon(0.01));
    }

    SECTION("and they match the distribution the car was built with")
    {
        const auto load = [&last](const Corner corner)
        {
            return last.corners[static_cast<std::size_t>(corner)].forces.tireVertical;
        };

        const auto front = load(Corner::FrontLeft) + load(Corner::FrontRight);
        const auto rear = load(Corner::RearLeft) + load(Corner::RearRight);

        REQUIRE(front / (front + rear) == Catch::Approx(0.60).epsilon(0.01));

        // And it is not leaning: a symmetric car on level ground loads both sides the same.
        //
        // To a tenth of a percent rather than to the bit, and the reason is worth knowing. Dropped
        // half a metre this car bottoms its suspension and the bodywork touches down for a few
        // ticks — which is correct, with 130 mm of static clearance — and the contact solver is
        // *sequential*: it walks the manifold point by point, so a perfectly symmetric contact does
        // not come out perfectly symmetric. The residual is under 0.002 degrees of roll and is a
        // property of the method rather than an error in it.
        REQUIRE(load(Corner::FrontLeft) == Catch::Approx(load(Corner::FrontRight)).epsilon(1e-3));
        REQUIRE(load(Corner::RearLeft) == Catch::Approx(load(Corner::RearRight)).epsilon(1e-3));
    }

    SECTION("it is sitting flat")
    {
        // Flat to well under a thousandth of a radian. Not exactly zero — see the note on
        // symmetry above.
        REQUIRE(last.telemetry.roll == Catch::Approx(0.0).margin(1e-3));
        REQUIRE(std::abs(last.telemetry.pitch) < 0.02);
    }
}

TEST_CASE("the vehicle tick is pure and deterministic", "[physics][vehicle][determinism]")
{
    // Criterion 12 at the level it finally matters: the same inputs from the same state must give
    // bit-identical state, and a saved state must continue identically to the one it was copied
    // from. Everything the brief asks for about netcode and rollback rests on these two.
    const JoltGuard jolt;

    const auto setup = placeholderSedan();
    REQUIRE(setup.has_value());

    const auto mesh = generateProvingGround(flatPlate());
    REQUIRE(mesh.has_value());
    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    const auto drive = [&](VehicleState& state, const int ticks)
    {
        for (auto tick = 0; tick < ticks; tick++)
        {
            auto input = VehicleInput{};
            input.steering = std::sin(static_cast<double>(tick) * 0.01) * 0.4;

            const auto stepped = stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), vehicleStep);
            REQUIRE(stepped.has_value());
        }
    };

    auto first = VehicleState{};
    first.chassis.position = glm::dvec3(0.0, 0.7, 20.0);
    auto second = first;

    drive(first, 600);
    drive(second, 600);

    REQUIRE(std::memcmp(&first, &second, sizeof(VehicleState)) == 0);

    SECTION("and a state restored from its own bytes carries on identically")
    {
        auto bytes = std::vector<unsigned char>(sizeof(VehicleState));
        std::memcpy(bytes.data(), &first, sizeof(VehicleState));

        auto restored = VehicleState{};
        std::memcpy(&restored, bytes.data(), sizeof(VehicleState));

        drive(first, 200);
        drive(restored, 200);

        REQUIRE(std::memcmp(&first, &restored, sizeof(VehicleState)) == 0);
    }
}
