// The steering torque's sign, pinned through the whole vehicle, because it was wrong twice in ways
// no other test could see. The tyre's aligning moment carried the inverse of the textbook's -t*Fy —
// worth ~2% of the chassis yaw moment, so every handling criterion passed, while at the rack it
// opposed the mechanical trail and left stage one slightly *pro-steer* at speed. The evdev backend
// then delivered every torque inverted (the kernel's 0x4000 direction is a force to the left, and
// positive demand here is a right turn), which cancelled the tyre fault while driving and exposed it
// parked: ease the wheel off centre with the car still and it pulled itself further over, a
// restoring torque delivered backwards. Two sign errors, visible only where they failed to cancel.
//
// Stage one's convention: positive steering torque is the same sense as positive demand, so a
// self-aligning tyre reports the *opposite* sign to the lock it is under — at a standstill and at
// speed both.

#include <array>
#include <cmath>
#include <cstddef>

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
using raceengine::ProvingGroundDescriptor;
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

// PlayerCar::publishRackTorque's assembly, restated: the same fields off the same tick.
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

} // namespace

TEST_CASE("the steering torque opposes the lock, parked and moving", "[physics][rack][ffb]")
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
    // Taken as the car states it. This used to force the sign by hand, because `rackTravelForSteer`
    // read its toe off a corner whose side label was mirrored and answered negative; the sheet was
    // correcting it with `steering.invert 0` and so was this line. Both are gone — see
    // `outboardSign`.
    const auto vehicle = built.value();
    REQUIRE(vehicle.rackTravelPerInput > 0.0);

    auto rack = SteeringRack{};
    rack.travelPerInput = vehicle.rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    SECTION("parked with the brake held, the wheel centres gently rather than pulling itself over")
    {
        // The reported failure, verbatim: sitting completely still, foot on the brake, ease the
        // wheel off centre. A torque with the lock's own sign is the positive feedback the driver
        // felt; a torque opposing it, of a couple of newton metres, is a parked tyre's twist.
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(vehicle, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }

        auto previousTravel = 0.0;
        auto settled = 0.0;

        for (auto step = 0; step < 1080; step++)
        {
            const auto time = static_cast<double>(step) * tick;
            auto input = VehicleInput{};
            input.brake = 1.0;
            input.steering = std::min(0.3, 0.6 * time);

            const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto travel = input.steering * vehicle.rackTravelPerInput;
            const auto velocity = (travel - previousTravel) / tick;
            previousTravel = travel;

            settled = rackTorqueOf(vehicle, state, stepped.value(), rack, travel, velocity).steeringTorque;
        }

        // Measured -2.04 N·m: opposing the +0.3 lock, and light. The brackets are loose on purpose —
        // what is being pinned is the sign and the order of magnitude, not a tuning.
        REQUIRE(settled < -0.2);
        REQUIRE(settled > -8.0);
    }

    SECTION("in a steady corner the wheel self-centres")
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);
        state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, 15.0);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            state.corners[index].wheelSpeed = 15.0 / 0.31;
        }

        auto previousTravel = 0.0;
        auto settled = 0.0;

        for (auto step = 0; step < 720; step++)
        {
            const auto time = static_cast<double>(step) * tick;
            auto input = VehicleInput{};
            input.steering = std::min(0.1, 0.4 * time);

            const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto travel = input.steering * vehicle.rackTravelPerInput;
            const auto velocity = (travel - previousTravel) / tick;
            previousTravel = travel;

            settled = rackTorqueOf(vehicle, state, stepped.value(), rack, travel, velocity).steeringTorque;
        }

        // Measured -10.2 N·m at 14.6 m/s under a +0.1 demand: mechanical and pneumatic trail
        // pulling the same way, about half the criterion-10 at-the-limit figure at half the load.
        // Before the aligning fix this settled at *+0.6* — the two trails cancelling to a slight
        // pull into the corner, which no absolute-value check could see.
        REQUIRE(settled < -2.0);
        REQUIRE(settled > -30.0);
    }
}
