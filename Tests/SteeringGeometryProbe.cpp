// A probe, not a gate. Hidden behind a dotted tag like the device probe and the shimmy probe, and
// run by hand: `./EngineTests "[.steering-geometry]"`.
//
// What it exists for is the two numbers the steering path is *supposed* to be checkable against, and
// which nothing printed before: the trail the hardpoints imply, and the steering ratio the linkage
// solves to. Criterion 10 asserted a magnitude window twenty-five times wider than its own measured
// value, which is a check that passes for any plausible scale factor — so the first thing needed was
// a number the geometry could be held to rather than a range it could sit anywhere inside.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::pinionRadius;
using raceengine::ProvingGroundDescriptor;
using raceengine::solveCorner;
using raceengine::SteeredCorner;
using raceengine::steeredCornerLimit;
using raceengine::SteeringRack;
using raceengine::steeringRackTorque;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::VehicleStep;

namespace
{

constexpr auto tick = 1.0 / 360.0;

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

// `SimulatedCar::publishRackTorque`'s assembly, restated so a probe can ask the same question the
// game asks without standing up a game.
[[nodiscard]] raceengine::RackTorque rackTorqueOf(const raceengine::VehicleSetup& setup, const VehicleState& state,
                                                  const VehicleStep& stepped, const SteeringRack& rack,
                                                  const double rackTravel, const double rackVelocity)
{
    const auto toBody = glm::conjugate(state.chassis.orientation);

    auto corners = std::array<SteeredCorner, steeredCornerLimit>{};

    for (auto index = std::size_t{0}; index < steeredCornerLimit; index++)
    {
        const auto& solution = stepped.corners[index];
        const auto& suspension = solution.suspension;
        const auto& hardpoints = setup.corners[index].hardpoints;

        const auto worldForce = glm::dvec3(0.0, solution.forces.tireVertical, 0.0) +
                                solution.contact.tyre.longitudinal * solution.contact.forward +
                                solution.contact.tyre.lateral * solution.contact.lateral;

        corners[index] = SteeredCorner{.lowerBallJoint = suspension.lowerBallJoint,
                                       .upperBallJoint = suspension.upperBallJoint,
                                       .steeringArm = suspension.steeringArm,
                                       .rackOuter = hardpoints.steeringRackOuter + glm::dvec3(rackTravel, 0.0, 0.0),
                                       .contactPatch = suspension.contactPatch,
                                       .patchNormal = toBody * solution.patch.normal,
                                       .tyreForce = toBody * worldForce,
                                       .aligningMoment = solution.contact.tyre.aligningMoment};
    }

    return steeringRackTorque(rack, corners, rackVelocity);
}

// Where the kingpin axis pierces the road, and what that says about the two levers a steered wheel
// has. Both are read off the *solved* geometry rather than authored, which is the whole point: move
// a hardpoint and these move with it.
struct GroundIntercept
{
    double mechanicalTrail = 0.0; // metres, positive when the patch trails the axis
    double scrubRadius = 0.0;     // metres, positive when the patch is outboard of the axis
    double kingpinInclination = 0.0;
    double caster = 0.0;
};

[[nodiscard]] GroundIntercept interceptOf(const glm::dvec3& lower, const glm::dvec3& upper, const glm::dvec3& patch,
                                          const double outboard)
{
    const auto span = upper - lower;
    // Parameterise along the axis to the patch's own height. The patch is where the road is, so the
    // road plane is the one that passes through it.
    const auto s = (patch.y - lower.y) / span.y;
    const auto intercept = lower + s * span;

    const auto axis = glm::normalize(span);

    return GroundIntercept{.mechanicalTrail = intercept.z - patch.z,
                           .scrubRadius = outboard * (patch.x - intercept.x),
                           // Lean of the axis in the front view, toward the car's centre.
                           .kingpinInclination = std::atan2(-outboard * axis.x, axis.y),
                           // Lean of the axis in the side view, top rearward is positive.
                           .caster = std::atan2(-axis.z, axis.y)};
}

} // namespace

TEST_CASE("what the Golf's front geometry implies about steering", "[.steering-geometry]")
{
    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    auto rack = SteeringRack{};
    rack.travelPerInput = car->rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    const auto radius = pinionRadius(rack);

    std::printf("\n--- rack ---\n");
    std::printf("travel per unit      %+.4f m\n", car->rackTravelPerInput);
    std::printf("lock to lock         %.1f deg\n", rack.lockToLockDegrees);
    std::printf("pinion radius        %.5f m\n", radius);

    for (auto index = std::size_t{0}; index < 2; index++)
    {
        const auto& hardpoints = car->corners[index].hardpoints;
        const auto outboard = raceengine::outboardSign(hardpoints.side);

        const auto solved = solveCorner(hardpoints, 0.0, 0.0);
        REQUIRE(solved.has_value());

        const auto geometry = interceptOf(solved->lowerBallJoint, solved->upperBallJoint, solved->contactPatch,
                                          outboard);

        // The steering ratio, by central difference on the solved toe. A millimetre of rack either
        // side: small enough that Ackermann has not moved much, large enough that the solve's own
        // iteration noise is nowhere near it.
        constexpr auto step = 0.001;
        const auto ahead = solveCorner(hardpoints, 0.0, step);
        const auto behind = solveCorner(hardpoints, 0.0, -step);
        REQUIRE(ahead.has_value());
        REQUIRE(behind.has_value());

        const auto steerPerTravel = (ahead->toe - behind->toe) / (2.0 * step);
        const auto ratio = 1.0 / (radius * std::abs(steerPerTravel));

        std::printf("\n--- corner %zu (%s, +x is the car's left) ---\n", index,
                    hardpoints.side == raceengine::CornerSide::Left ? "left" : "right");
        std::printf("contact patch        %+.4f %+.4f %+.4f\n", solved->contactPatch.x, solved->contactPatch.y,
                    solved->contactPatch.z);
        std::printf("mechanical trail     %+.4f m\n", geometry.mechanicalTrail);
        std::printf("scrub radius         %+.4f m\n", geometry.scrubRadius);
        std::printf("kingpin inclination  %+.2f deg\n", glm::degrees(geometry.kingpinInclination));
        std::printf("caster               %+.2f deg\n", glm::degrees(geometry.caster));
        std::printf("d(steer)/d(rack)     %+.4f rad/m\n", steerPerTravel);
        std::printf("steering ratio       %.2f :1 (rim per road wheel)\n", ratio);
    }
}

TEST_CASE("what an axle at the limit puts through the rack, unassisted", "[.steering-geometry]")
{
    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    auto rack = SteeringRack{};
    rack.travelPerInput = car->rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    // A front axle at the limit. This car's front pair carries about 8 kN and a semislick makes
    // rather more than its own load in side force at peak.
    constexpr auto sideForce = 4000.0;
    constexpr auto verticalLoad = 4000.0;
    constexpr auto pneumaticTrail = 0.025;

    auto corners = std::array<SteeredCorner, 2>{};
    auto totalMechanical = 0.0;

    for (auto index = std::size_t{0}; index < 2; index++)
    {
        const auto& hardpoints = car->corners[index].hardpoints;
        const auto solved = solveCorner(hardpoints, 0.0, 0.0);
        REQUIRE(solved.has_value());

        const auto outboard = raceengine::outboardSign(hardpoints.side);
        const auto geometry = interceptOf(solved->lowerBallJoint, solved->upperBallJoint, solved->contactPatch,
                                          outboard);
        totalMechanical += geometry.mechanicalTrail;

        corners[index] = SteeredCorner{.lowerBallJoint = solved->lowerBallJoint,
                                       .upperBallJoint = solved->upperBallJoint,
                                       .steeringArm = solved->steeringArm,
                                       .rackOuter = hardpoints.steeringRackOuter,
                                       .contactPatch = solved->contactPatch,
                                       .patchNormal = glm::dvec3(0.0, 1.0, 0.0),
                                       .tyreForce = glm::dvec3(sideForce, verticalLoad, 0.0),
                                       .aligningMoment = -sideForce * pneumaticTrail};
    }

    const auto answer = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);
    REQUIRE(answer.finite);

    std::printf("\n--- axle at 4 kN a corner, unassisted, this car's own rack ---\n");
    std::printf("kingpin torque       %+.2f  %+.2f N.m\n", answer.kingpinTorque[0], answer.kingpinTorque[1]);
    std::printf("tyre force at rack   %+.2f N\n", answer.tyreForce);
    std::printf("torque at the rim    %+.3f N.m\n", answer.steeringTorque);
    std::printf("mean mechanical trail %+.4f m (pair sum %+.4f)\n", totalMechanical / 2.0, totalMechanical);

    // The same, with only the lateral force, which is the term the trail identity is about.
    for (auto& corner : corners)
    {
        corner.tyreForce = glm::dvec3(sideForce, 0.0, 0.0);
        corner.aligningMoment = 0.0;
    }

    const auto lateralOnly = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);
    std::printf("lateral only, no Mz  %+.3f N.m at the rim\n", lateralOnly.steeringTorque);

    // And with only the vertical load, which is what should cancel between two equally loaded sides.
    for (auto& corner : corners)
    {
        corner.tyreForce = glm::dvec3(0.0, verticalLoad, 0.0);
    }

    const auto verticalOnly = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);
    std::printf("vertical only        %+.3f N.m at the rim\n", verticalOnly.steeringTorque);
}

TEST_CASE("what the unassisted rack asks of a driver, parked and at speed", "[.steering-geometry]")
{
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

    auto rack = SteeringRack{};
    rack.travelPerInput = vehicle.rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    SECTION("parked, brake held, quasi-static sweep to full lock")
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(vehicle, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }

        std::printf("\n--- parked, brake held, unassisted ---\n");
        std::printf("  demand   rim deg   rack mm   tyre N   rack N   rim N.m\n");

        auto previousTravel = 0.0;
        auto nextReport = 0.1;

        // Eight seconds to full lock, which is slow enough that the rack's own damping term is a
        // rounding error and what is being read is the tyre.
        for (auto step = 0; step < 2880; step++)
        {
            const auto time = static_cast<double>(step) * tick;
            auto input = VehicleInput{};
            input.brake = 1.0;
            input.steering = std::min(1.0, time / 8.0);

            const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto travel = input.steering * vehicle.rackTravelPerInput;
            const auto velocity = (travel - previousTravel) / tick;
            previousTravel = travel;

            const auto answer = rackTorqueOf(vehicle, state, stepped.value(), rack, travel, velocity);

            if (input.steering >= nextReport - 1e-9)
            {
                std::printf("  %+.2f    %7.1f   %7.2f   %8.1f  %8.1f   %+7.3f\n", input.steering,
                            input.steering * rack.lockToLockDegrees * 0.5, travel * 1000.0, answer.tyreForce,
                            answer.rackForce, answer.steeringTorque);
                nextReport += 0.1;
            }
        }
    }

    SECTION("rolling, quasi-static sweep until the fronts give up")
    {
        for (const auto speed : {8.0, 20.0, 35.0})
        {
            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);
            state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                state.corners[index].wheelSpeed = speed / 0.31;
            }

            std::printf("\n--- rolling at %.0f m/s, unassisted ---\n", speed);
            std::printf("  demand   lat g   front N   tyre N   rack N   rim N.m\n");

            auto totalMass = vehicle.unsprungMass();
            for (const auto& component : vehicle.sprung)
            {
                totalMass += component.mass;
            }

            auto previousTravel = 0.0;
            auto nextReport = 0.05;

            for (auto step = 0; step < 3600; step++)
            {
                const auto time = static_cast<double>(step) * tick;
                auto input = VehicleInput{};
                input.steering = std::min(0.5, time / 20.0);

                const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), tick);
                REQUIRE(stepped.has_value());

                const auto travel = input.steering * vehicle.rackTravelPerInput;
                const auto velocity = (travel - previousTravel) / tick;
                previousTravel = travel;

                const auto answer = rackTorqueOf(vehicle, state, stepped.value(), rack, travel, velocity);

                if (input.steering >= nextReport - 1e-9)
                {
                    auto lateral = 0.0;
                    auto frontLateral = 0.0;
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        lateral += stepped->corners[index].contact.tyre.lateral;
                        if (index < steeredCornerLimit)
                        {
                            frontLateral += stepped->corners[index].contact.tyre.lateral;
                        }
                    }

                    std::printf("  %+.2f   %+.3f  %8.1f  %8.1f  %8.1f   %+7.3f\n", input.steering,
                                lateral / (totalMass * 9.80665), frontLateral, answer.tyreForce, answer.rackForce,
                                answer.steeringTorque);
                    nextReport += 0.05;
                }
            }
        }
    }
}
