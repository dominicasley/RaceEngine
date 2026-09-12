// Stage 2 of docs/kerb-contact-brief.md: the kerb-contact path on a car, on the proving ground.
//
// The Golf, on the two vertical step features the ground grew for this, with the path switched on —
// and, first, with it switched on over the roads the goldens drive, where it must change nothing to
// the bit. Every number a fixture measures is printed with WARN so a run reports what it saw whether
// or not a criterion held; the criteria are the brief's and are stated where they are asserted.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bodyToWorld;
using raceengine::bringUpJolt;
using raceengine::Corner;
using raceengine::cornerCount;
using raceengine::CornerForces;
using raceengine::CornerSide;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto tyreHalfWidth = 0.1125;
constexpr auto degrees = 57.29577951308232;

constexpr auto frontLeft = static_cast<std::size_t>(Corner::FrontLeft);
constexpr auto frontRight = static_cast<std::size_t>(Corner::FrontRight);

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

[[nodiscard]] VehicleSetup golf(const bool kerbContact)
{
    auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    built->kerbContact = kerbContact;

    return std::move(built).value();
}

[[nodiscard]] PhysicsWorld worldOf(const ProvingGroundDescriptor& descriptor)
{
    const auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());
    auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    return std::move(world).value();
}

// A flat road of the stated size, a quarter-metre grid.
[[nodiscard]] ProvingGroundDescriptor road(const double length, const double width)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = width;
    descriptor.cellSize = 0.25;
    descriptor.features = {};

    return descriptor;
}

// The Golf's own settle, then a speed: four seconds at rest from the design height, then every
// wheel spun to match the road.
void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double startZ)
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

// Everything a fixture reads off one tick, for one corner and the car.
struct Sample
{
    std::uint32_t obstacleCandidates = 0;
    std::uint32_t obstacleCount = 0;
    double obstacleForce = 0.0;
    double obstacleElevation = 0.0;
    double load = 0.0;
    double hubHeight = 0.0;
    double hubX = 0.0;
    double patchHeight = 0.0;
    double damperVelocity = 0.0;
    double speed = 0.0;
    double positionZ = 0.0;
    double positionX = 0.0;
    double centreOfMassHeight = 0.0;
    double rightY = 0.0;
    // The kerb contacts' own roll moment about the car's forward axis, from their push directions
    // and their lever arms about the centre of mass: negative rolls the +x side down.
    double kerbRollSense = 0.0;
    std::array<double, cornerCount> loads{};
    std::array<bool, cornerCount> loaded{};
    bool contactsBelowCentre = true;
    bool bodyTouching = false;
};

[[nodiscard]] Sample sampleTick(const VehicleSetup& setup, VehicleState& state, const VehicleInput& input,
                                const std::array<double, cornerCount>& drive, const PhysicsWorld& world,
                                const std::size_t corner)
{
    // The chassis the tick was solved against, so the hub's world position pairs with the
    // suspension solve that produced it rather than with the state one tick on.
    const auto before = state.chassis;

    const auto stepped = stepVehicle(setup, state, input, drive, world, tick);
    REQUIRE(stepped.has_value());

    const auto& solution = stepped->corners[corner];

    auto sample = Sample{};
    sample.obstacleCandidates = solution.obstacleCandidates;
    sample.obstacleCount = solution.obstacleCount;
    sample.obstacleForce = solution.obstacleNormalForce;
    sample.obstacleElevation = solution.obstacleAxisElevation * degrees;
    sample.load = solution.forces.tireVertical;
    const auto hub = bodyToWorld(before, solution.suspension.wheelCentre);
    sample.hubHeight = hub.y;
    sample.hubX = hub.x;
    sample.patchHeight = solution.patch.inContact ? solution.patch.centre.y : 0.0;
    sample.damperVelocity = solution.damperVelocity;
    sample.speed = glm::length(state.chassis.linearVelocity);
    sample.positionZ = state.chassis.position.z;
    sample.positionX = state.chassis.position.x;
    sample.centreOfMassHeight = before.position.y;
    sample.rightY = (state.chassis.orientation * glm::dvec3(1.0, 0.0, 0.0)).y;
    sample.bodyTouching = !stepped->contacts.points.empty();

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& each = stepped->corners[index];
        sample.loads[index] = each.forces.tireVertical;
        sample.loaded[index] = each.forces.tireVertical > 0.0 || each.obstacleNormalForce > 0.0;

        for (auto slot = std::uint32_t{0}; slot < each.obstacleCount; slot++)
        {
            const auto& contact = each.obstacles[slot];
            if (contact.point.y >= before.position.y)
            {
                sample.contactsBelowCentre = false;
            }

            sample.kerbRollSense += glm::cross(contact.point - before.position, contact.axis).z;
        }
    }

    return sample;
}

} // namespace

TEST_CASE("the kerb-contact path is inert on flat, banked and chamfered road", "[physics][kerb][fixture][inertness]")
{
    // The first thing the switch has to prove, and the brief's own condition for it going on: over
    // the roads the goldens drive, the path changes nothing. Not "nearly nothing" — the state and the
    // corner forces are compared as bytes, tick for tick, because thirty seconds of launch amplify an
    // ulp into a parity failure. The positive control is the candidate count: the cylinder saw the
    // road on every one of those ticks, and the rule kept none of it.
    const JoltGuard jolt;

    struct Course
    {
        const char* name;
        ProvingGroundDescriptor descriptor;
        double steering;
        int ticks;
    };

    auto banked = road(200.0, 30.0);
    // Eleven and a half degrees of bank, eased in over the band: steeper than anything the circuit
    // carries, and still well inside the fifteen degree cone.
    banked.camberAngle = 0.20;
    banked.features = {Feature{.kind = FeatureKind::Camber, .from = 30.0, .to = 110.0}};

    // The E2 crossing, exactly as `the imported car crosses a kerb continuously` drives it: the
    // 9.46 degree chamfer, half-mounted at a shallow angle.
    auto chamfer = road(200.0, 30.0);
    chamfer.cellSize = 0.10;
    chamfer.kerbInnerEdge = 0.60;
    chamfer.features = {Feature{.kind = FeatureKind::Kerb, .from = 40.0, .to = 160.0}};

    const auto courses = std::array<Course, 3>{
        Course{.name = "flat", .descriptor = road(200.0, 30.0), .steering = 0.0, .ticks = 1800},
        Course{.name = "banked", .descriptor = banked, .steering = 0.0, .ticks = 3600},
        Course{.name = "chamfer",
               .descriptor = chamfer,
               .steering = 0.01 * outboardSign(CornerSide::Right),
               .ticks = 3600},
    };

    for (const auto& course : courses)
    {
        CAPTURE(course.name);

        const auto world = worldOf(course.descriptor);

        // One fingerprint per tick per run: the whole state's bytes, then every corner's forces.
        struct Run
        {
            std::vector<unsigned char> fingerprint;
            std::uint64_t candidates = 0;
            std::uint64_t survivors = 0;
            double seconds = 0.0;
        };

        const auto run = [&](const bool on)
        {
            const auto setup = golf(on);
            auto state = VehicleState{};
            settle(setup, state, world, 8.0, 20.0);

            auto input = VehicleInput{};
            input.steering = course.steering;

            auto result = Run{};
            result.fingerprint.reserve(static_cast<std::size_t>(course.ticks) *
                                       (sizeof(VehicleState) + cornerCount * sizeof(CornerForces)));

            const auto started = std::chrono::steady_clock::now();
            for (auto step = 0; step < course.ticks; step++)
            {
                const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick);
                REQUIRE(stepped.has_value());

                const auto* stateBytes = reinterpret_cast<const unsigned char*>(&state);
                result.fingerprint.insert(result.fingerprint.end(), stateBytes, stateBytes + sizeof(VehicleState));

                for (const auto& corner : stepped->corners)
                {
                    const auto* forceBytes = reinterpret_cast<const unsigned char*>(&corner.forces);
                    result.fingerprint.insert(result.fingerprint.end(), forceBytes, forceBytes + sizeof(CornerForces));
                    result.candidates += corner.obstacleCandidates;
                    result.survivors += corner.obstacleCount;
                }
            }
            result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

            return result;
        };

        const auto off = run(false);
        const auto on = run(true);

        WARN(course.name << ": " << course.ticks << " ticks, off " << off.seconds * 1e6 / course.ticks
                         << " us/tick, on " << on.seconds * 1e6 / course.ticks << " us/tick; the cylinder saw "
                         << on.candidates << " road triangles and the rule kept " << on.survivors);

        // The positive control first: a path that never queried would be inert for the wrong reason.
        REQUIRE(off.candidates == 0);
        REQUIRE(on.candidates > static_cast<std::uint64_t>(course.ticks));
        REQUIRE(on.survivors == 0);

        // Then the claim.
        REQUIRE(off.fingerprint.size() == on.fingerprint.size());
        REQUIRE(std::memcmp(off.fingerprint.data(), on.fingerprint.data(), off.fingerprint.size()) == 0);
    }
}

TEST_CASE("a 150 mm step met head-on at 33 kph is a face contact and not a road step",
          "[physics][kerb][fixture][mount]")
{
    // The finding, replayed on the proving ground: GCP's kerb at the speed the seat trace mounted it.
    // What the bottom grid alone does here is 30 kN in one tick and the wheel into its stop at four
    // metres a second; what the brief asks of the path is the list asserted below.
    const JoltGuard jolt;

    auto descriptor = road(120.0, 20.0);
    descriptor.stepHeight = 0.15;
    descriptor.features = {Feature{.kind = FeatureKind::StepAcross, .from = 40.0, .to = 60.0}};

    const auto world = worldOf(descriptor);
    const auto setup = golf(true);

    auto state = VehicleState{};
    settle(setup, state, world, 33.0 / 3.6, 20.0);

    auto samples = std::vector<Sample>{};
    for (auto step = 0; step < 2160; step++)
    {
        samples.push_back(sampleTick(setup, state, VehicleInput{}, noDriveTorque, world, frontLeft));
        REQUIRE(std::isfinite(state.chassis.position.y));
    }

    // Where the face was met.
    auto firstTouch = std::size_t{0};
    while (firstTouch < samples.size() && samples[firstTouch].obstacleCount == 0)
    {
        firstTouch++;
    }
    REQUIRE(firstTouch < samples.size());
    REQUIRE(firstTouch > 60);

    // A second of the crossing, from first touch.
    const auto windowEnd = std::min(samples.size(), firstTouch + 360);

    auto peakForce = 0.0;
    auto peakTick = firstTouch;
    auto peakHubVelocity = 0.0;
    auto worstDamperStep = 0.0;
    auto peakElevation = 0.0;
    auto longestUnloaded = std::size_t{0};
    auto unloadedRun = std::size_t{0};

    for (auto index = firstTouch; index < windowEnd; index++)
    {
        const auto& at = samples[index];
        const auto& previous = samples[index - 1];

        if (at.obstacleForce > peakForce)
        {
            peakForce = at.obstacleForce;
            peakTick = index;
        }

        peakHubVelocity = std::max(peakHubVelocity, (at.hubHeight - previous.hubHeight) / tick);
        worstDamperStep = std::max(worstDamperStep, std::abs(at.damperVelocity - previous.damperVelocity));
        peakElevation = std::max(peakElevation, at.obstacleElevation);

        const auto allLoaded =
            std::all_of(at.loaded.begin(), at.loaded.end(), [](const bool loaded) { return loaded; });
        unloadedRun = allLoaded ? 0 : unloadedRun + 1;
        longestUnloaded = std::max(longestUnloaded, unloadedRun);
    }

    const auto speedBefore = samples[firstTouch - 1].speed;
    const auto speedAfter = samples[windowEnd - 1].speed;
    const auto onTop = samples[windowEnd - 1].patchHeight;

    // The crossing tick by tick, for the record: what the seat trace could only show as one tick.
    auto table = std::string{};
    for (auto index = firstTouch - 2; index < std::min(samples.size(), firstTouch + 40); index++)
    {
        const auto& at = samples[index];
        table += "\n  t+" + std::to_string(static_cast<long long>(index) - static_cast<long long>(firstTouch)) +
                 ": face " + std::to_string(static_cast<int>(at.obstacleForce)) + " N at " +
                 std::to_string(static_cast<int>(at.obstacleElevation)) + " deg, load " +
                 std::to_string(static_cast<int>(at.load)) + " N, hub +" +
                 std::to_string(static_cast<int>((at.hubHeight - samples[index - 1].hubHeight) / tick * 1000.0)) +
                 " mm/s, damper " + std::to_string(static_cast<int>(at.damperVelocity * 1000.0)) + " mm/s, speed " +
                 std::to_string(at.speed).substr(0, 5) + ", patch " +
                 std::to_string(static_cast<int>(at.patchHeight * 1000.0)) + " mm";
    }
    WARN("the crossing:" << table);

    WARN("first touch at tick " << firstTouch << " with the face " << samples[firstTouch].obstacleElevation
                                << " deg above the horizon; peak face force " << peakForce << " N after "
                                << (peakTick - firstTouch) << " ticks, elevation up to " << peakElevation
                                << " deg; peak hub rise " << peakHubVelocity << " m/s; worst damper step "
                                << worstDamperStep << " m/s per tick; speed " << speedBefore << " -> " << speedAfter
                                << " m/s; longest spell with a wheel unloaded " << longestUnloaded
                                << " ticks; front-left patch ends " << onTop << " m up");

    // The brief's five criteria, as CHECKs so that one run reports every verdict, and the two
    // sanity conditions as REQUIREs. **Four of the five are red at the shipped numbers**
    // (2026-09-08, docs/known-red.md): the tick table above is what the path does at a 150 mm
    // face at 33 kph with the tread rate at the edge and the placed 40000 N.s/m bump stop under
    // it, and the brief's own rule is that neither the stop nor the rate is tuned against this.
    //
    // Load rises from zero over ten or more ticks. A tick is 25 mm of travel at this speed, so the
    // first touching tick is already two centimetres deep; the criterion is read as ticks from
    // first touch to the peak.
    CHECK(peakTick - firstTouch >= 10);
    // The car loses speed.
    CHECK(speedAfter < speedBefore - 0.1);
    // Peak wheel vertical velocity under 1.5 m/s. **A rigid 0.3186 m wheel's hub rises 0.150 m
    // over the 0.270 m between first touch and the hub passing the edge, which at 9.2 m/s is a mean
    // of 5 m/s** — the criterion the brief's Progress flagged before the fixture ran; the number
    // above is what the tyre's compliance makes of it.
    CHECK(peakHubVelocity < 1.5);
    // No damper-velocity reversal over 2 m/s in one tick. The reversal is the bump stop, at t+8.
    CHECK(worstDamperStep < 2.0);
    // All four wheels loaded again within 0.3 s. The car is launched and flies for most of a
    // second, as the seat trace's body did.
    CHECK(longestUnloaded <= 108);
    // And it did mount: the front-left patch is on the kerb top a second later, and the car did
    // not pass through.
    REQUIRE(onTop > 0.10);
}

TEST_CASE("a sideways slide into a 150 mm face is a sidewall contact below the centre of mass",
          "[physics][kerb][fixture][slide]")
{
    // The tripped rollover's first half: a car sliding sideways meets a kerb face with its sidewall,
    // the push arrives below the centre of mass, and the body rolls towards the kerb. Whether it goes
    // over is the second half and is not asked here.
    const JoltGuard jolt;

    auto descriptor = road(60.0, 20.0);
    descriptor.stepHeight = 0.15;
    // The kerb is on +x, which is the car's left; the left tyres' outer sidewalls stand at about
    // 0.88 m, so there is 0.22 m of slide before the face.
    descriptor.kerbInnerEdge = 1.10;
    descriptor.features = {Feature{.kind = FeatureKind::StepAlong, .from = 10.0, .to = 50.0}};

    const auto world = worldOf(descriptor);
    const auto setup = golf(true);

    auto state = VehicleState{};
    settle(setup, state, world, 0.0, 30.0);
    state.chassis.linearVelocity = glm::dvec3(3.0, 0.0, 0.0);

    auto samples = std::vector<Sample>{};
    for (auto step = 0; step < 720; step++)
    {
        samples.push_back(sampleTick(setup, state, VehicleInput{}, noDriveTorque, world, frontLeft));
        REQUIRE(std::isfinite(state.chassis.position.x));
    }

    auto firstTouch = std::size_t{0};
    while (firstTouch < samples.size() && samples[firstTouch].obstacleCount == 0)
    {
        firstTouch++;
    }
    REQUIRE(firstTouch < samples.size());

    auto peakForce = 0.0;
    auto belowCentre = true;
    auto rollTowardKerb = true;
    auto rightYAtTouch = samples[firstTouch].rightY;
    auto rightYMin = rightYAtTouch;
    auto furthestHub = -std::numeric_limits<double>::max();
    auto lastTouch = firstTouch;

    for (auto index = firstTouch; index < samples.size(); index++)
    {
        const auto& at = samples[index];
        furthestHub = std::max(furthestHub, at.hubX);

        if (at.obstacleCount > 0)
        {
            lastTouch = index;
            peakForce = std::max(peakForce, at.obstacleForce);
            belowCentre = belowCentre && at.contactsBelowCentre;
            rollTowardKerb = rollTowardKerb && at.kerbRollSense < 0.0;
            rightYMin = std::min(rightYMin, at.rightY);
        }
    }

    // How far the sidewall went in, which is what the placeholder lateral rate is worth at this
    // speed: the hub's furthest x plus the section half-width, against the face.
    const auto sidewallCompression = furthestHub + tyreHalfWidth - descriptor.kerbInnerEdge;

    WARN("slide: first touch at tick " << firstTouch << " at " << samples[firstTouch].speed << " m/s, held "
                                       << (lastTouch - firstTouch + 1) << " ticks, peak sidewall push " << peakForce
                                       << " N; body right-vector y " << rightYAtTouch << " -> " << rightYMin
                                       << "; sidewall pressed " << sidewallCompression * 1000.0
                                       << " mm into the face; hub x " << furthestHub << " against a face at "
                                       << descriptor.kerbInnerEdge);

    REQUIRE(peakForce > 0.0);
    // The lateral force acts below the centre of mass.
    REQUIRE(belowCentre);
    // Roll moment toward the kerb: every kerb contact's own moment rolls the +x side down, and the
    // body's right vector dips while it is pushed. The far-side tyre loads cannot arbitrate here —
    // a 3 m/s sideways skid has the rear far-side wheel at zero load before the face is reached.
    REQUIRE(rollTowardKerb);
    REQUIRE(rightYMin < rightYAtTouch);
    // No pass-through: the hub never crosses the face plane, and the car is still on its wheels.
    // The sidewall's compression is reported rather than bounded, because it is the placeholder
    // lateral rate's answer and the rate is the ledger's to move.
    REQUIRE(furthestHub < descriptor.kerbInnerEdge);
    REQUIRE(std::abs(state.chassis.orientation.x) < 0.2);
    REQUIRE(std::abs(state.chassis.orientation.z) < 0.2);
}

TEST_CASE("a driven wheel climbs a 100 mm kerb from rest", "[physics][kerb][fixture][climb]")
{
    // What the friction is for: a front-wheel-drive car parked against a kerb, given its torque,
    // gets its front wheels onto it. Without friction at the face the push-out is radial and the
    // wheel can only be shoved back.
    const JoltGuard jolt;

    auto descriptor = road(60.0, 20.0);
    descriptor.stepHeight = 0.10;
    descriptor.features = {Feature{.kind = FeatureKind::StepAcross, .from = 30.0, .to = 40.0}};

    const auto world = worldOf(descriptor);
    const auto setup = golf(true);

    // Two centimetres short of the face with the front tyres' leading edge.
    const auto frontAxle = setup.corners[frontLeft].hardpoints.wheelCentre.z;
    const auto startZ = 30.0 - 0.02 - tyreRadius - frontAxle;

    auto state = VehicleState{};
    settle(setup, state, world, 0.0, startZ);

    // First gear's worth at the front hubs and nothing at the rear.
    auto drive = std::array<double, cornerCount>{};
    drive[frontLeft] = 1500.0;
    drive[frontRight] = 1500.0;

    auto samples = std::vector<Sample>{};
    auto climbedAt = std::size_t{0};
    auto peakForce = 0.0;
    auto peakSpin = 0.0;
    for (auto step = 0; step < 1440; step++)
    {
        // A rev limiter, because a constant torque on a spinning wheel is not an engine: first gear
        // runs out at about 45 rad/s at the hub.
        for (const auto corner : {frontLeft, frontRight})
        {
            drive[corner] = state.corners[corner].wheelSpeed < 45.0 ? 1500.0 : 0.0;
        }

        samples.push_back(sampleTick(setup, state, VehicleInput{}, drive, world, frontLeft));
        REQUIRE(std::isfinite(state.chassis.position.z));

        peakForce = std::max(peakForce, samples.back().obstacleForce);
        peakSpin = std::max(peakSpin, state.corners[frontLeft].wheelSpeed);
        if (climbedAt == 0 && samples.back().patchHeight > 0.09)
        {
            climbedAt = samples.size();
        }
    }

    WARN("climb: front patch on the kerb top after "
         << climbedAt << " ticks; peak face force " << peakForce << " N; peak front wheel speed " << peakSpin
         << " rad/s; car advanced " << (state.chassis.position.z - startZ) << " m");

    REQUIRE(peakForce > 0.0);
    REQUIRE(climbedAt > 0);
    REQUIRE(state.chassis.position.z > startZ + 0.3);
}

TEST_CASE("a wall above hub height is the chassis collider's and not the tyre's", "[physics][kerb][fixture][wall]")
{
    // The Golf pushed sideways into the barrier with the path on: the bodywork's contact solver
    // stops it, and the tyre's cylinder query keeps nothing on any wheel on any tick, because a push
    // at hub height is a wall.
    const JoltGuard jolt;

    auto descriptor = road(200.0, 20.0);
    descriptor.barrierX = 1.10;
    descriptor.barrierHeight = 0.9;
    descriptor.features = {Feature{.kind = FeatureKind::Barrier, .from = 20.0, .to = 180.0}};

    const auto world = worldOf(descriptor);
    const auto setup = golf(true);

    auto state = VehicleState{};
    settle(setup, state, world, 0.0, 100.0);
    state.chassis.linearVelocity = glm::dvec3(2.0, 0.0, 0.0);

    auto touched = false;
    auto survivors = std::uint64_t{0};
    auto candidates = std::uint64_t{0};
    for (auto step = 0; step < 1800; step++)
    {
        const auto sample = sampleTick(setup, state, VehicleInput{}, noDriveTorque, world, frontLeft);
        touched = touched || sample.bodyTouching;
        survivors += sample.obstacleCount;
        candidates += sample.obstacleCandidates;
    }

    WARN("wall: bodywork touched " << (touched ? "yes" : "no") << "; the front-left cylinder saw " << candidates
                                   << " triangles and the rule kept " << survivors);

    REQUIRE(touched);
    REQUIRE(candidates > 0);
    REQUIRE(survivors == 0);
}
