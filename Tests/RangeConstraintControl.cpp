// The byte-inert control for the linkage range constraint (docs/range-constraint-brief.md, 2026-09-08
// later still): the shipped Golf on a flat plate, PLACED at its rest height rather than dropped onto
// it, through five scenarios that keep every corner strictly inside both authored range limits at
// every tick — a ring-down, two wheel hops, the characterisation's 0.5-pedal stop (the rear's peak
// extension is 1.08 mm short of its droop clamp there, so this is the control that matters), and a
// 0.30 steering step at 20 m/s. The state is printed as hexadecimal floats. Run on the build before
// the change and the build after; the two prints must not differ by a character.
//
// The fixture proves its own precondition: each scenario prints every corner's closest approach to
// each of its limits and requires both to be positive, so a print that came out of a corner that
// touched a limit cannot be mistaken for a control.
//
// **Not the settle drop.** Every whole-car fixture in this project starts by dropping the car from
// 0.52 m, and the wheels hang on their droop limits through the bounce — so the drop is exactly the
// case the constraint changes, and a control that started with it would prove nothing.
//
// Used only the API that existed before the range-constraint change, so that it compiled against both
// builds; since 2026-09-08 latest of all it also reads one test-local knob,
// `OSR_CONTROL_FRAME_ACCELERATION=off`, which states `VehicleSetup::frameAcceleration` off on the copy
// it runs (docs/frame-acceleration-brief.md) — the placed car's body accelerates through every one of
// these scenarios, so the print with the switch ON is the positive control and the print with it OFF
// is the byte-inert one against the build before that change. Since 2026-09-08 latest of all (d) a
// second, `OSR_CONTROL_KINEMATIC_REACTION=off`, states `VehicleSetup::kinematicReaction` off the same
// way (docs/chassis-kinematic-reaction-brief.md): the placed car's wheels swing on their arcs through
// every scenario, so ON moves this print and OFF is the byte-inert control against the build before.
//
// `./EngineTests "[.range-constraint-control]"`.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::seedTyreGasPressures;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;

// The car placed on the plate: the body's origin 8 mm below the road so the tyres start close to
// their static penetration and the transient is millimetres of heave rather than a drop. The centre
// of mass is read the way the sandbox reads it, off one inert tick.
void place(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double z)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 1.0, z);
    REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, 1e-6).has_value());

    const auto centreOfMass = state.chassis.centreOfMass;
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, centreOfMass.y - 0.008, z);
}

// Place, ring down for two seconds, seed the cavity gas at ideal, then give the car its speed.
void placeAndSettle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
                    const double z)
{
    place(setup, state, world, z);
    for (auto step = 0; step < 720; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    seedTyreGasPressures(setup, state);

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / setup.corners.front().hardpoints.wheelRadius;
    }
}

// The characterisation's proportional-integral speed hold on the front wheels.
struct SpeedHold
{
    double target = 0.0;
    double integral = 0.0;

    [[nodiscard]] std::array<double, cornerCount> torques(const VehicleSetup& setup, const VehicleState& state)
    {
        if (target <= 0.0)
        {
            return noDriveTorque;
        }

        const auto error = target - glm::length(state.chassis.linearVelocity);
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                              setup.corners.front().hardpoints.wheelRadius / 2.0;

        return {perWheel, perWheel, 0.0, 0.0};
    }
};

void printState(const int at, const VehicleState& state, const raceengine::VehicleStep& step)
{
    const auto& chassis = state.chassis;
    std::printf("  tick %5d position %a %a %a velocity %a %a %a momentum %a %a %a orientation %a %a %a %a\n", at,
                chassis.position.x, chassis.position.y, chassis.position.z, chassis.linearVelocity.x,
                chassis.linearVelocity.y, chassis.linearVelocity.z, chassis.angularMomentum.x,
                chassis.angularMomentum.y, chassis.angularMomentum.z, chassis.orientation.w, chassis.orientation.x,
                chassis.orientation.y, chassis.orientation.z);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = state.corners[index];
        const auto& solution = step.corners[index];
        std::printf("    %s angle %a rate %a spin %a damper %a bump %a droop %a Fz %a Q %a\n",
                    cornerAbbreviation(static_cast<Corner>(index)), corner.wishboneAngle, corner.wishboneRate,
                    corner.wheelSpeed, solution.forces.damper, solution.forces.bumpStop, solution.forces.droopStop,
                    solution.forces.tireVertical, solution.generalisedForce);
    }
}

// Run one scenario, print the state at the sampled ticks, and require that no corner reached a limit.
void scenario(const char* title, const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world,
              const int ticks, SpeedHold& hold,
              const std::function<VehicleInput(int)>& inputAt = [](int) { return VehicleInput{}; })
{
    std::printf("\n=== %s ===\n", title);

    auto toDroop = std::array<double, cornerCount>{1e9, 1e9, 1e9, 1e9};
    auto toBump = std::array<double, cornerCount>{1e9, 1e9, 1e9, 1e9};

    for (auto step = 1; step <= ticks; step++)
    {
        const auto input = inputAt(step);
        const auto stepped = stepVehicle(setup, state, input, hold.torques(setup, state), world, tick);
        REQUIRE(stepped.has_value());

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& hardpoints = setup.corners[index].hardpoints;
            const auto angle = state.corners[index].wishboneAngle;
            toDroop[index] = std::min(toDroop[index], angle - hardpoints.droopAngle);
            toBump[index] = std::min(toBump[index], hardpoints.bumpAngle - angle);
        }

        if (step == 1 || step == 2 || step == 10 || step == 100 || step == 360 || step == 720 || step == ticks)
        {
            printState(step, state, stepped.value());
        }
    }

    std::printf("  closest approach to the limits, mrad:");
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        std::printf("  %s droop %.3f bump %.3f", cornerAbbreviation(static_cast<Corner>(index)), toDroop[index] * 1000.0,
                    toBump[index] * 1000.0);
        REQUIRE(toDroop[index] > 0.0);
        REQUIRE(toBump[index] > 0.0);
    }
    std::printf("\n");
}

} // namespace

TEST_CASE("the placed car inside its travel, bit for bit", "[.range-constraint-control]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    REQUIRE(bringUpJolt().has_value());

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 1600.0;
    descriptor.width = 1600.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto setup = golf.value();
    if (const auto* env = std::getenv("OSR_CONTROL_FRAME_ACCELERATION"); env != nullptr && std::string(env) == "off")
    {
        setup.frameAcceleration = false;
    }
    if (const auto* env = std::getenv("OSR_CONTROL_KINEMATIC_REACTION"); env != nullptr && std::string(env) == "off")
    {
        setup.kinematicReaction = false;
    }
    std::printf("frameAcceleration %d\n", setup.frameAcceleration);
    std::printf("kinematicReaction %d\n", setup.kinematicReaction);

    auto state = VehicleState{};

    {
        place(setup, state, world.value(), 800.0);
        auto hold = SpeedHold{};
        scenario("placed at rest height, 3 s ring-down", setup, state, world.value(), 1080, hold);
    }

    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        for (const auto rate : {1.0, 3.0})
        {
            placeAndSettle(setup, state, world.value(), 0.0, 800.0);
            state.corners[index].wishboneRate = rate;
            auto hold = SpeedHold{};
            const auto title = std::string(index == 0 ? "front" : "rear") + " wheel hop, " + std::to_string(rate) +
                               " rad/s, settled first";
            scenario(title.c_str(), setup, state, world.value(), 1080, hold);
        }
    }

    {
        placeAndSettle(setup, state, world.value(), 25.0, 800.0);
        auto hold = SpeedHold{};
        scenario("0.5-pedal stop from 25 m/s, nobody intervening (the rear's peak extension is 1.08 mm short of its clamp)",
                 setup, state, world.value(), 1440, hold,
                 [](const int step)
                 {
                     auto input = VehicleInput{};
                     input.brake = step > 180 ? 0.5 : 0.0;
                     return input;
                 });
    }

    {
        placeAndSettle(setup, state, world.value(), 20.0, 800.0);
        auto hold = SpeedHold{.target = 20.0};
        scenario("0.30 steering step at 20 m/s, speed held", setup, state, world.value(), 1800, hold,
                 [](const int step)
                 {
                     auto input = VehicleInput{};
                     input.steering = step > 180 ? 0.30 : 0.0;
                     return input;
                 });
    }

    tearDownJolt();
}
