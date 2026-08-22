// **Every fixture in this suite is a straight aimed along world +z, and that is a structural hole.**
//
// Two telemetry channels were found carrying world-frame components under body-frame names, and the
// reason nothing caught it is that on a straight aimed along +z the two frames agree. It is not a
// hypothetical: the coastdown criterion recovered plausible `Crr` and `CdA` partly *because* it is a
// straight, so a world-frame longitudinal reading was accidentally the right one. A whole class of
// frame confusions is invisible to this suite by construction.
//
// So this closes the class rather than the instances. The same manoeuvre is driven at four headings
// a quarter turn and an eighth apart, and every quantity that is supposed to be a property of the
// *car* is required to come out the same. Anything that changes with which way the car happens to be
// pointing is frame-confused, whatever it is called and whoever added it.
//
// It is the same move as splitting the parity gate by scene: structural rather than a promise. A new
// channel is covered the day it is added, because it is covered by the loop rather than by somebody
// remembering to write a case for it.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TelemetryFrame;
using raceengine::VehicleInput;
using raceengine::VehicleState;

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

// One named number off a telemetry frame, so the comparison below can be a loop over names rather
// than a list of assertions somebody has to remember to extend.
//
// `scale` is what this channel is worth when it matters — a full-lock rim, a corner's share of the
// car's weight, a hard braking g. The comparison is relative to it rather than to the reading, and
// that is not a softening: several of these settle near zero in steady state, where a purely
// relative tolerance asks for agreement to a fraction of the residual bobbing and catches nothing
// but noise. What this test exists to find is an axis swap or a factor, both of which are the size
// of the channel itself.
struct Channel
{
    // A `std::string` and not a `const char*`: the per-corner names are built here, and holding
    // pointers into a vector that is still being appended to is a dangling read the moment it
    // reallocates. It cost one confusing failure report with an empty channel name.
    std::string name;
    double value;
    double scale;
};

[[nodiscard]] std::vector<Channel> channelsOf(const TelemetryFrame& frame)
{
    auto channels = std::vector<Channel>{
        // The car's own accelerations. These are the two that were wrong.
        {"acceleration long", frame.acceleration.z, 10.0},
        {"acceleration lat", frame.acceleration.x, 10.0},
        {"acceleration vert", frame.acceleration.y, 10.0},
        // Attitude and rates. Yaw is deliberately absent: it is *which way the car is pointing*, so
        // it is the one channel that must differ, and it is checked as such below.
        {"pitch", frame.pitch, 0.1},
        {"roll", frame.roll, 0.1},
        {"yaw rate", frame.yawRate, 0.5},
        {"pitch rate", frame.pitchRate, 0.5},
        {"roll rate", frame.rollRate, 0.5},
        // Driver inputs, and the rim angle derived from one of them.
        {"steering demand", frame.steering, 1.0},
        {"steering wheel angle", frame.steeringWheelAngle, 6.6},
        {"throttle", frame.throttle, 1.0},
        {"brake", frame.brake, 1.0},
        // **Inert in this fixture and here anyway.** These three are filled by whoever stepped the
        // driveline and this fixture steps none, so they read zero at every heading and the
        // comparison below is 0 == 0. That is honest coverage of the *structure* rather than of the
        // numbers: none of the three has a frame to be confused about, and the day this fixture
        // gains a driveline they are already in the loop rather than waiting for somebody to
        // remember them. What they are worth in absolute terms is checked in `TelemetryUnitsProbe`,
        // which is where the `Engine RPM` factor of 9.55 should have been caught.
        {"clutch", frame.clutch, 1.0},
        {"gear", static_cast<double>(frame.gear), 6.0},
        {"engine speed", frame.engineSpeed, 300.0},
        // Height above the ground is a world quantity that a heading cannot change.
        {"height", frame.position.y, 0.5},
        {"vertical speed", frame.velocity.y, 1.0},
        {"speed", glm::length(frame.velocity), 20.0},
        // How the car is sitting on its springs. A property of the car and its load transfer, so
        // pointing it a different way must not move it — and until 2026-08-21 these reached no
        // channel at all, which is why nothing could have noticed either way.
        {"ride height front", frame.rideHeightFront, 0.1},
        {"ride height rear", frame.rideHeightRear, 0.1},
    };

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto tag = std::string(raceengine::cornerAbbreviation(static_cast<raceengine::Corner>(index)));
        const auto& wheel = frame.wheels[index];

        for (const auto& [suffix, value, scale] : std::array<std::tuple<const char*, double, double>, 11>{
                 {{"load", wheel.verticalLoad, 5000.0},
                  {"slip ratio", wheel.slipRatio, 0.1},
                  {"slip angle", wheel.slipAngle, 0.2},
                  {"force long", wheel.forceLongitudinal, 5000.0},
                  {"force lat", wheel.forceLateral, 5000.0},
                  {"aligning moment", wheel.aligningMoment, 100.0},
                  {"susp pos", wheel.suspensionTravel, 0.05},
                  {"damper vel", wheel.damperVelocity, 0.5},
                  {"camber", wheel.camber, 0.05},
                  // How much of the patch grid is touching. On flat ground this is the whole grid
                  // whichever way the car points, so what it is watching for here is a sampling
                  // pattern that is laid out in *world* axes rather than in the wheel's — which
                  // would put a different number of samples on the ground at 45 degrees, and is
                  // exactly the class of fault this suite exists to close.
                  {"contact samples", static_cast<double>(wheel.contactingSamples), 25.0},
                  // How tilted the carrying part of the patch is. On flat ground it is not zero —
                  // the tread curves away fore and aft, so the leading and trailing rows sit about
                  // ten millimetres shallower than the middle one — and that residual is the point:
                  // it is a property of the *tyre's own shape*, so it must read the same at every
                  // heading. A sampling grid laid out in world axes would swing it, and so would a
                  // patch frame built from the world's forward rather than the wheel's.
                  {"patch depth spread", wheel.patchDepthSpread, 0.02}}})
        {
            channels.push_back(Channel{tag + " " + suffix, value, scale});
        }
    }

    return channels;
}

} // namespace

TEST_CASE("a car's own channels do not depend on which way it is pointing", "[physics][telemetry][frame]")
{
    const JoltGuard jolt;

    // Square and centred, so a car pointed any way has the same ground under it for the whole run.
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 600.0;
    descriptor.width = 600.0;
    descriptor.cellSize = 3.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto vehicle = built.value();

    // A manoeuvre with something on every axis: braking into a turn, so there is longitudinal
    // acceleration, lateral acceleration, roll, pitch and four different wheel loads at once. A
    // straight-line run would agree at every heading for the wrong reason.
    //
    // **Deliberately well inside the limit.** A car sliding is a chaotic run, and two chaotic runs
    // started a quarter turn apart on a triangulated grid genuinely diverge — which would be read
    // here as a frame fault. The linear region is where "the same manoeuvre" means the same thing.
    const auto driveFacing = [&](const double heading)
    {
        auto state = VehicleState{};
        state.chassis.orientation = glm::angleAxis(heading, glm::dvec3(0.0, 1.0, 0.0));
        // The middle of the ground, whichever way it then goes.
        state.chassis.position = glm::dvec3(0.0, 0.52, 300.0);
        state.chassis.linearVelocity = state.chassis.orientation * glm::dvec3(0.0, 0.0, 20.0);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            state.corners[index].wheelSpeed = 20.0 / 0.31;
        }

        // **Averaged over the last second rather than read off the final tick.** The acceleration
        // channels are a difference of two velocities across one tick of contact impulses, so a
        // single tick of them is the noisiest thing in the frame — and a car crossing a triangulated
        // grid at a different angle meets its seams at different moments, which is a real difference
        // between two runs and is not a frame fault. Everything asserted here is a steady-state
        // quantity, so a steady-state measurement is what it should be compared as.
        auto accumulated = std::vector<Channel>{};
        auto samples = 0;

        for (auto step = 0; step < 1080; step++)
        {
            const auto time = static_cast<double>(step) * tick;

            auto input = VehicleInput{};
            input.steering = std::min(0.08, 0.2 * time);
            input.brake = time > 1.0 ? 0.20 : 0.0;

            const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            if (step >= 720)
            {
                auto channels = channelsOf(stepped->telemetry);

                if (accumulated.empty())
                {
                    accumulated = channels;
                }
                else
                {
                    for (auto index = std::size_t{0}; index < accumulated.size(); index++)
                    {
                        accumulated[index].value += channels[index].value;
                    }
                }

                samples++;
            }
        }

        for (auto& channel : accumulated)
        {
            channel.value /= static_cast<double>(samples);
        }

        return accumulated;
    };

    const auto referenceChannels = driveFacing(0.0);

    // A quarter turn either way, an eighth, and a heading that is not a multiple of anything — the
    // last one because axis-aligned headings can hide a fault that swaps two axes cleanly.
    for (const auto heading : {1.5707963267948966, -1.5707963267948966, 0.7853981633974483, 2.4})
    {
        const auto rotated = driveFacing(heading);
        REQUIRE(rotated.size() == referenceChannels.size());

        for (auto index = std::size_t{0}; index < rotated.size(); index++)
        {
            CAPTURE(heading, referenceChannels[index].name, referenceChannels[index].value, rotated[index].value);

            // Three percent of whichever is larger: the reading or what the channel is worth when it
            // matters. The margin is not tight on purpose — the ground is a triangulated grid and a
            // car crossing it at a different angle meets its seams at different moments, so the two
            // runs are genuinely not the same arithmetic. What is being caught here is a factor or
            // an axis swap, and both are the size of the channel rather than the size of the noise:
            // the two faults that prompted this test moved `G Force Long` from -6.44 to +4.01.
            const auto scale = std::max(std::abs(referenceChannels[index].value), referenceChannels[index].scale);
            REQUIRE(std::abs(rotated[index].value - referenceChannels[index].value) < 0.03 * scale);
        }
    }
}

TEST_CASE("and yaw is the one channel that must change with it", "[physics][telemetry][frame]")
{
    // The control for the case above. If every channel were invariant, the loop would pass just as
    // happily against a fixture that had stopped rotating the car at all — so one channel has to be
    // shown moving, and yaw is the one that means "which way is it pointing".
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 600.0;
    descriptor.width = 600.0;
    descriptor.cellSize = 3.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto yawFacing = [&](const double heading)
    {
        auto state = VehicleState{};
        state.chassis.orientation = glm::angleAxis(heading, glm::dvec3(0.0, 1.0, 0.0));
        state.chassis.position = glm::dvec3(0.0, 0.52, 300.0);

        const auto stepped = stepVehicle(built.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(stepped.has_value());

        return stepped->telemetry.yaw;
    };

    REQUIRE(yawFacing(0.0) == Catch::Approx(0.0).margin(1e-6));
    REQUIRE(yawFacing(1.5707963267948966) == Catch::Approx(1.5707963267948966).margin(1e-6));
    REQUIRE(yawFacing(-0.7853981633974483) == Catch::Approx(-0.7853981633974483).margin(1e-6));
}

TEST_CASE("and where it went turns with it, without changing how far", "[physics][telemetry][frame]")
{
    // The other two channels that must *not* be invariant, and the reason they need saying out loud.
    //
    // `CoG X` and `CoG Z` join the trace with the telemetry expansion, and they are world-frame
    // positions: a car driven the same way pointing a different direction ends up somewhere else,
    // and that is correct. So the loop above would be wrong to hold them and a test that simply
    // omitted them would leave nobody stating what they *are* — which is how a channel ends up
    // quietly carrying a body-frame offset under a world-frame name.
    //
    // What is a property of the car is how far it travelled and how the path is shaped. Both are
    // asserted here: the displacement's length is the same at every heading, and its direction turns
    // by exactly the heading. That pins the pair together — a swap of the two components passes
    // neither, and so does one of them silently reading a body-frame coordinate.
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 600.0;
    descriptor.width = 600.0;
    descriptor.cellSize = 3.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto vehicle = built.value();

    // The start is the same point for every heading, so the displacement is the whole difference.
    const auto start = glm::dvec3(0.0, 0.52, 300.0);

    const auto travelFacing = [&](const double heading)
    {
        auto state = VehicleState{};
        state.chassis.orientation = glm::angleAxis(heading, glm::dvec3(0.0, 1.0, 0.0));
        state.chassis.position = start;
        state.chassis.linearVelocity = state.chassis.orientation * glm::dvec3(0.0, 0.0, 20.0);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            state.corners[index].wheelSpeed = 20.0 / 0.31;
        }

        auto travelled = glm::dvec3(0.0);

        for (auto step = 0; step < 1080; step++)
        {
            const auto time = static_cast<double>(step) * tick;

            auto input = VehicleInput{};
            input.steering = std::min(0.08, 0.2 * time);
            input.brake = time > 1.0 ? 0.20 : 0.0;

            const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            travelled = stepped->telemetry.position - start;
        }

        return travelled;
    };

    const auto reference = travelFacing(0.0);

    // It has to have gone somewhere, or every claim below is a claim about zero.
    REQUIRE(glm::length(glm::dvec2(reference.x, reference.z)) > 50.0);

    for (const auto heading : {1.5707963267948966, -1.5707963267948966, 0.7853981633974483, 2.4})
    {
        const auto rotated = travelFacing(heading);

        CAPTURE(heading, reference.x, reference.z, rotated.x, rotated.z);

        // How far, in the ground plane. A property of the car and the manoeuvre.
        const auto referenceGround = glm::dvec2(reference.x, reference.z);
        const auto rotatedGround = glm::dvec2(rotated.x, rotated.z);

        REQUIRE(glm::length(rotatedGround) ==
                Catch::Approx(glm::length(referenceGround)).epsilon(0.03));

        // And which way. The chassis yaws about +y, and `TelemetryFrame::yaw` is `atan2(forward.x,
        // forward.z)` — so a heading of theta takes (x, z) to (x·cos + z·sin, z·cos - x·sin), which
        // is what turning the reference displacement by the same angle means here. Written out
        // rather than reached for through a matrix, because the whole point is to state the
        // convention rather than to inherit it from whatever glm would have done.
        const auto expected = glm::dvec2(referenceGround.x * std::cos(heading) + referenceGround.y * std::sin(heading),
                                         referenceGround.y * std::cos(heading) - referenceGround.x * std::sin(heading));

        // Against the length rather than against each component: one of the two passes through zero
        // at a quarter turn, where a relative tolerance asks for agreement to a fraction of nothing.
        REQUIRE(glm::length(rotatedGround - expected) < 0.03 * glm::length(referenceGround));
    }
}
