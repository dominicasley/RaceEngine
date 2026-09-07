// The street furniture, and the one question that decides whether any of it is real: does a prop
// stand until it is hit hard enough, and does it come free when it is.
//
// **There is no constraint anywhere in this and that is the design.** Jolt has no breakable joint,
// and the usual answer — a fixed constraint per prop whose accumulated lambda is read every tick —
// would be three and a half thousand constraints recovering a number this engine already has: it
// resolves the car's contacts itself, on purpose, so the impulse a break threshold is quoted against
// is an output of the solver rather than something to be reconstructed from one.
//
// So the seam under test is `applyPropImpulses`: hand the world an impulse and a point, and it
// decides. What is deliberately *not* here is a car — a manifold is the thing that produces these
// impulses in the game and it needs a vehicle, a track and a collision to happen, none of which say
// anything more about the threshold than a number applied by hand does.

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::BreakableProp;
using raceengine::bringUpJolt;
using raceengine::ConvexCollider;
using raceengine::generateProvingGround;
using raceengine::PhysicsWorld;
using raceengine::PropImpulse;
using raceengine::PropTransform;
using raceengine::PropVelocity;
using raceengine::ProvingGroundDescriptor;
using raceengine::tearDownJolt;

using Catch::Matchers::WithinAbs;

namespace
{

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

// The simulation's own tick, because the thresholds are forces and a force is an impulse over one of
// these. A test that used a different step would be testing a different threshold.
constexpr auto tick = 1.0 / 360.0;

// A lamp column's box, eight metres of it, stated about its own centre — which is where the exporter
// puts a body's origin, and is what lets the inertia tensor go in with no parallel-axis shift.
[[nodiscard]] std::vector<glm::dvec3> columnHull(const glm::dvec3& origin)
{
    auto points = std::vector<glm::dvec3>();
    for (const auto x : {-0.1, 0.1})
    {
        for (const auto y : {-4.0, 4.0})
        {
            for (const auto z : {-0.1, 0.1})
            {
                points.emplace_back(x, y, z);
            }
        }
    }

    static_cast<void>(origin);

    return points;
}

// Grand City Parkway's own `prop_0000`, near enough: 108.8 kg, and a tensor that is a slender column
// about its own centre rather than the solid prism its hull describes.
[[nodiscard]] BreakableProp lampColumn(const glm::dvec3& origin)
{
    auto prop = BreakableProp{};
    prop.hulls.push_back(ConvexCollider{.points = columnHull(origin), .origin = origin, .surface = 0});
    prop.mass = 108.8;
    prop.inertia = glm::dmat3(541.9, 0.0, 0.0, 0.0, 0.486, 0.0, 0.0, 0.0, 541.9);
    prop.breakForce = 18000.0;
    prop.breakTorque = 22000.0;

    return prop;
}

[[nodiscard]] PhysicsWorld flatWorldWith(const BreakableProp& prop)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 200.0;
    descriptor.width = 200.0;
    descriptor.cellSize = 5.0;
    descriptor.features = {};

    auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());

    // A world with props needs at least one collider, because the props take the surface the
    // colliders were given. One kerbstone well out of the way is the least that satisfies it.
    const auto kerb =
        ConvexCollider{.points = columnHull(glm::dvec3(0.0)), .origin = glm::dvec3(80.0, 0.0, 80.0), .surface = 0};

    const auto props = std::vector<BreakableProp>{prop};
    const auto colliders = std::vector<ConvexCollider>{kerb};

    auto world = PhysicsWorld::create(ground.value(), colliders, props);
    REQUIRE(world.has_value());

    return std::move(world).value();
}

} // namespace

TEST_CASE("a prop stands until it is hit hard enough, and then it is loose", "[physics][props]")
{
    const JoltGuard jolt;

    // Standing on the ground: eight metres of column about its own centre puts the origin at four.
    const auto origin = glm::dvec3(0.0, 4.0, 0.0);
    auto world = flatWorldWith(lampColumn(origin));

    auto loose = std::vector<PropTransform>();

    SECTION("nothing is loose in a world nothing has hit")
    {
        world.freeProps(loose);
        REQUIRE(loose.empty());

        // And stepping it changes nothing, because there is nothing to integrate.
        world.step(tick);
        world.freeProps(loose);
        REQUIRE(loose.empty());
    }

    SECTION("an impulse under both thresholds leaves it standing")
    {
        // 10 N·s over one tick is 3600 N against an 18 kN base, and applied at the origin it exerts
        // no moment at all — so neither threshold is approached.
        const auto nudge =
            std::vector<PropImpulse>{PropImpulse{.prop = 0, .impulse = glm::dvec3(10.0, 0.0, 0.0), .point = origin}};

        world.applyPropImpulses(nudge, tick);
        world.step(tick);

        world.freeProps(loose);
        REQUIRE(loose.empty());
    }

    SECTION("an impulse over the force threshold releases it")
    {
        // 100 N·s over one tick is 36 kN, twice what the base holds.
        const auto hit =
            std::vector<PropImpulse>{PropImpulse{.prop = 0, .impulse = glm::dvec3(100.0, 0.0, 0.0), .point = origin}};

        world.applyPropImpulses(hit, tick);

        world.freeProps(loose);
        REQUIRE(loose.size() == 1);
        REQUIRE(loose.front().prop == 0);
    }

    SECTION("the torque threshold releases it on its own")
    {
        // **The discriminating case, and the reason a break tests two numbers rather than one.** A
        // blow near the top of a column is a couple: 30 N·s three metres up is 10.8 kN of force,
        // well under the 18 kN base — and 32.4 kN·m of moment against a 22 kN·m base, which is what
        // actually snaps a lighting column at its plate.
        const auto high = origin + glm::dvec3(0.0, 3.0, 0.0);
        const auto blow =
            std::vector<PropImpulse>{PropImpulse{.prop = 0, .impulse = glm::dvec3(30.0, 0.0, 0.0), .point = high}};

        REQUIRE(glm::length(glm::dvec3(30.0, 0.0, 0.0)) / tick < 18000.0);

        world.applyPropImpulses(blow, tick);

        world.freeProps(loose);
        REQUIRE(loose.size() == 1);
    }

    SECTION("a loose prop is integrated and a standing one is not")
    {
        const auto hit = std::vector<PropImpulse>{PropImpulse{
            .prop = 0, .impulse = glm::dvec3(400.0, 0.0, 0.0), .point = origin + glm::dvec3(0.0, 3.5, 0.0)}};

        world.applyPropImpulses(hit, tick);

        world.freeProps(loose);
        REQUIRE(loose.size() == 1);
        const auto released = loose.front();

        for (auto step = 0; step < 360; step++)
        {
            world.step(tick);
        }

        world.freeProps(loose);
        REQUIRE(loose.size() == 1);

        // A second of it: pushed along +x by the blow, and rotated out of upright by the couple. The
        // assertion is that it *moved*, not where it got to — where a felled column lands is a
        // question for the seat and not for a threshold test.
        const auto travelled = glm::length(loose.front().position - released.position);
        REQUIRE(travelled > 0.5);

        const auto upright = loose.front().orientation * glm::dvec3(0.0, 1.0, 0.0);
        REQUIRE(upright.y < 0.999);
    }
}

TEST_CASE("the break impulse is the first shove and not every tick's", "[physics][props]")
{
    const JoltGuard jolt;

    // **What this is guarding.** Before the contact solver took a second body, `applyPropImpulses`
    // was the only way a prop ever moved, so it pushed one on every tick it was named. A loose prop
    // now takes its momentum through the solver's own two-body exchange, which is applied as a
    // velocity just before the world steps — so pushing it here as well would count one collision
    // twice, and a car resting against a bin would accelerate it for ever.
    const auto origin = glm::dvec3(0.0, 4.0, 0.0);
    const auto hit =
        std::vector<PropImpulse>{PropImpulse{.prop = 0, .impulse = glm::dvec3(400.0, 0.0, 0.0), .point = origin}};

    auto quiet = flatWorldWith(lampColumn(origin));
    auto shoved = flatWorldWith(lampColumn(origin));

    quiet.applyPropImpulses(hit, tick);
    shoved.applyPropImpulses(hit, tick);

    for (auto step = 0; step < 120; step++)
    {
        quiet.step(tick);

        shoved.applyPropImpulses(hit, tick);
        shoved.step(tick);
    }

    auto loose = std::vector<PropTransform>();

    quiet.freeProps(loose);
    REQUIRE(loose.size() == 1);
    const auto left = loose.front().position;

    shoved.freeProps(loose);
    REQUIRE(loose.size() == 1);
    const auto right = loose.front().position;

    // One 400 N.s impulse on 108.8 kg is 3.68 m/s and a third of a second of it is about a metre.
    // A hundred and twenty-one of them would be 441 m/s and a hundred and fifty metres, so the
    // half-metre tolerance is nowhere near either boundary — it is there for Jolt's own threading,
    // not for the quantity under test.
    for (auto axis = 0; axis < 3; axis++)
    {
        REQUIRE_THAT(right[axis], WithinAbs(left[axis], 0.5));
    }
}

TEST_CASE("a velocity change reaches a loose prop and never a standing one", "[physics][props]")
{
    const JoltGuard jolt;

    const auto origin = glm::dvec3(0.0, 4.0, 0.0);
    const auto hit =
        std::vector<PropImpulse>{PropImpulse{.prop = 0, .impulse = glm::dvec3(400.0, 0.0, 0.0), .point = origin}};

    auto loose = std::vector<PropTransform>();

    SECTION("a standing prop is left where it is")
    {
        // A static body has no motion properties for a velocity to be added to, and Jolt asserts on
        // one. The guard lives in the backend rather than in the caller because the caller cannot
        // see a break that happened after its manifold was built.
        auto world = flatWorldWith(lampColumn(origin));

        world.applyPropVelocities(std::vector<PropVelocity>{
            PropVelocity{.prop = 0, .linear = glm::dvec3(30.0, 0.0, 0.0), .angular = glm::dvec3(0.0, 2.0, 0.0)}});
        world.step(tick);

        world.freeProps(loose);
        REQUIRE(loose.empty());
    }

    SECTION("a loose one takes it")
    {
        auto still = flatWorldWith(lampColumn(origin));
        auto pushed = flatWorldWith(lampColumn(origin));

        still.applyPropImpulses(hit, tick);
        pushed.applyPropImpulses(hit, tick);

        still.freeProps(loose);
        REQUIRE(loose.size() == 1);
        const auto start = loose.front().position;

        // Along +z, which is an axis the break impulse put nothing into — so what arrives there
        // arrived through this call and through nothing else.
        pushed.applyPropVelocities(std::vector<PropVelocity>{
            PropVelocity{.prop = 0, .linear = glm::dvec3(0.0, 0.0, 12.0), .angular = glm::dvec3(0.0)}});

        for (auto step = 0; step < 60; step++)
        {
            still.step(tick);
            pushed.step(tick);
        }

        still.freeProps(loose);
        const auto quietPosition = loose.front().position;

        pushed.freeProps(loose);
        const auto movedPosition = loose.front().position;

        REQUIRE_THAT(quietPosition.z - start.z, WithinAbs(0.0, 0.05));
        REQUIRE(movedPosition.z - start.z > 1.0);
    }
}

TEST_CASE("releasing a prop tests nothing and simply lets it go", "[physics][props]")
{
    const JoltGuard jolt;

    // **The seam the car actually uses now.** `releaseBrokenProps` decides, inside the vehicle tick
    // and before `resolveContacts` runs, which anchors cannot hold the collision that is about to
    // happen; this call only carries that decision to the world. There is deliberately no threshold
    // in it — one decision in one place is what stops the module and the backend disagreeing about a
    // prop sitting exactly on its own limit.
    const auto origin = glm::dvec3(0.0, 4.0, 0.0);
    auto world = flatWorldWith(lampColumn(origin));

    auto loose = std::vector<PropTransform>();

    SECTION("a released prop is free and is integrated")
    {
        const auto released = std::vector<std::uint32_t>{0};

        world.releaseProps(released);

        world.freeProps(loose);
        REQUIRE(loose.size() == 1);
        REQUIRE(loose.front().prop == 0);

        // Freed and left alone it stands where it was, because nothing gave it any momentum. That is
        // the whole difference from the old path, which handed it an impulse sized to stop a car.
        const auto start = loose.front().position;

        world.applyPropVelocities(std::vector<PropVelocity>{
            PropVelocity{.prop = 0, .linear = glm::dvec3(0.0, 0.0, 4.0), .angular = glm::dvec3(0.0)}});

        for (auto step = 0; step < 90; step++)
        {
            world.step(tick);
        }

        world.freeProps(loose);

        // A quarter of a second at 4 m/s is a metre, and it is the only thing that moved it.
        REQUIRE_THAT(loose.front().position.z - start.z, WithinAbs(1.0, 0.15));
    }

    SECTION("releasing one twice is harmless and an unknown index is ignored")
    {
        const auto released = std::vector<std::uint32_t>{0, 0, 99};

        world.releaseProps(released);
        world.releaseProps(released);

        world.freeProps(loose);
        REQUIRE(loose.size() == 1);
    }
}
