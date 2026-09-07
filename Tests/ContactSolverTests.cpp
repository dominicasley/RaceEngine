// The contact solver's own arithmetic: two bodies exchanging momentum, and friction that has a
// direction.
//
// **A separate file from BreakablePropTests.cpp, whose subject genuinely differs.** That one asks
// whether a prop stands until it is hit hard enough and says in its own header that there is
// deliberately no car in it. This one asks what `resolveContacts` computes, and there is deliberately
// no *world* in it: every case below is a hand-built manifold and a hand-built body, so a failure
// names a term rather than a scene.
//
// The first test is the one that matters most and reads as the least interesting. Both frame gates
// are blessed on a track that carries no props, so every contact in a golden capture is against an
// immovable second body — and the whole safety argument for adding one is that the arithmetic then
// reduces *exactly*, not nearly, to the one-body solver this replaced. The gates check that at scale
// and say only that something moved. This says which term.

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::ContactBody;
using raceengine::ContactManifold;
using raceengine::ContactMaterial;
using raceengine::ContactPoint;
using raceengine::releaseBrokenProps;
using raceengine::resolveContacts;
using raceengine::RigidBodyState;
using raceengine::tangentBasis;

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{

// The simulation's own tick, because the Baumgarte bias is a penetration divided by one of these.
constexpr auto tick = 1.0 / 360.0;

// A chassis with a real inertia tensor, sitting off the origin and turning, so that every term in
// the effective mass is live rather than multiplied by a convenient zero. The Golf's, near enough.
[[nodiscard]] RigidBodyState chassis()
{
    auto state = RigidBodyState{};

    state.mass = 1400.0;
    state.inertia = glm::dmat3(560.0, 0.0, 0.0, 0.0, 2400.0, 0.0, 0.0, 0.0, 2100.0);
    state.inverseInertia = glm::inverse(state.inertia);
    state.position = glm::dvec3(3.0, 0.55, -7.0);
    state.orientation = glm::normalize(glm::dquat(0.94, 0.05, 0.31, -0.11));
    state.linearVelocity = glm::dvec3(-4.0, -0.7, 2.5);
    state.angularMomentum = glm::dvec3(120.0, -260.0, 55.0);

    return state;
}

// Two points on a face, off the centre of mass in every axis, with penetration on one of them so
// that the bias term is exercised too.
[[nodiscard]] ContactManifold wallManifold()
{
    auto manifold = ContactManifold{};

    manifold.points.push_back(ContactPoint{.position = glm::dvec3(1.2, 0.3, -6.4),
                                           .normal = glm::normalize(glm::dvec3(0.82, 0.12, -0.56)),
                                           .penetration = 0.021});
    manifold.points.push_back(ContactPoint{.position = glm::dvec3(1.35, 0.9, -7.7),
                                           .normal = glm::normalize(glm::dvec3(0.82, 0.12, -0.56)),
                                           .penetration = 0.004});

    return manifold;
}

// Exact equality, term by term, and named so that a failure says which one moved rather than that
// two structs differ.
void requireIdentical(const RigidBodyState& left, const RigidBodyState& right)
{
    for (auto axis = 0; axis < 3; axis++)
    {
        REQUIRE(left.position[axis] == right.position[axis]);
        REQUIRE(left.linearVelocity[axis] == right.linearVelocity[axis]);
        REQUIRE(left.angularMomentum[axis] == right.angularMomentum[axis]);
    }

    REQUIRE(left.orientation.w == right.orientation.w);
    REQUIRE(left.orientation.x == right.orientation.x);
    REQUIRE(left.orientation.y == right.orientation.y);
    REQUIRE(left.orientation.z == right.orientation.z);
    REQUIRE(left.mass == right.mass);
}

// A point-mass body, so that a friction case has one term in it and its answer can be written down
// rather than measured. A zero inverse inertia is a legitimate state — `worldInverseInertia` is
// R·0·Rt — and it is the only way to take rotation out of the effective mass without also taking out
// the arm.
[[nodiscard]] RigidBodyState pointMass(const glm::dvec3& velocity)
{
    auto state = RigidBodyState{};

    state.mass = 1400.0;
    state.inertia = glm::dmat3(1.0);
    state.inverseInertia = glm::dmat3(0.0);
    state.position = glm::dvec3(0.0, 1.0, 0.0);
    state.linearVelocity = velocity;

    return state;
}

// One flat contact under that body, with no penetration and no restitution to muddy the impulse.
[[nodiscard]] ContactManifold flatContact()
{
    auto manifold = ContactManifold{};

    manifold.points.push_back(
        ContactPoint{.position = glm::dvec3(0.0, 0.0, 0.0), .normal = glm::dvec3(0.0, 1.0, 0.0), .penetration = 0.0});

    return manifold;
}

[[nodiscard]] ContactMaterial plainMaterial()
{
    auto material = ContactMaterial{};

    material.restitution = 0.0;
    material.friction = 0.6;

    return material;
}

} // namespace

TEST_CASE("an immovable second body is bit-identical to no second body at all", "[physics][contact]")
{
    const auto material = ContactMaterial{};

    // Once with no table at all, which is every manifold built before a second body existed and
    // every one a fixture writes by hand.
    auto without = chassis();
    auto absent = wallManifold();
    resolveContacts(without, absent, material, tick);

    // And once with the table `collideBody` builds for a track that carries no props: one explicit
    // entry, immovable, named by every point.
    auto with = chassis();
    auto stated = wallManifold();
    stated.bodies.push_back(ContactBody{});
    resolveContacts(with, stated, material, tick);

    requireIdentical(without, with);

    // And once more the way a *standing* prop arrives: its own entry, named by every point, carrying
    // a centre of mass metres from the car and still an infinite effective mass. This is the shipped
    // path on a track with street furniture on it, and it has to reduce the same way — a prop that is
    // anchored to the ground is a building until the tick it breaks.
    auto anchored = chassis();
    auto standing = wallManifold();
    standing.bodies.push_back(ContactBody{});
    standing.bodies.push_back(ContactBody{.prop = 7, .centre = glm::dvec3(1.4, 3.9, -7.1)});
    for (auto& point : standing.points)
    {
        point.prop = 7;
        point.body = 1;
    }
    resolveContacts(anchored, standing, material, tick);

    requireIdentical(without, anchored);
    REQUIRE(standing.bodies[1].deltaLinear == glm::dvec3(0.0));
    REQUIRE(standing.bodies[1].deltaAngular == glm::dvec3(0.0));

    // And the impulses that produced it, because two states could agree by both being wrong.
    REQUIRE(absent.points.size() == stated.points.size());
    for (auto index = std::size_t{0}; index < absent.points.size(); index++)
    {
        REQUIRE(absent.points[index].normalImpulse == stated.points[index].normalImpulse);
        REQUIRE(absent.points[index].tangentImpulse1 == stated.points[index].tangentImpulse1);
        REQUIRE(absent.points[index].tangentImpulse2 == stated.points[index].tangentImpulse2);
    }

    // The immovable body is left exactly where it was, which is what "infinite mass" has to mean if
    // the write-back is ever going to be safe to apply unconditionally.
    REQUIRE(stated.bodies.front().deltaLinear == glm::dvec3(0.0));
    REQUIRE(stated.bodies.front().deltaAngular == glm::dvec3(0.0));
}

TEST_CASE("a car and a loose prop exchange momentum along the normal", "[physics][contact]")
{
    // Head-on, and arranged so that both arms lie along the normal: the cross products vanish, the
    // effective mass is the two inverse masses alone, and the answer can be written down.
    //
    //   j = (v_rel) / (1/m_car + 1/m_prop)
    //
    // A rotation would make this a true statement about a solver rather than about arithmetic, and
    // the two-point wall case above is where the arms are live.
    const auto material = plainMaterial();

    const auto build = [&material](const double propMass)
    {
        auto state = pointMass(glm::dvec3(-5.0, 0.0, 0.0));
        state.position = glm::dvec3(1.0, 0.0, 0.0);

        auto manifold = ContactManifold{};
        manifold.bodies.push_back(ContactBody{});
        manifold.bodies.push_back(ContactBody{.prop = 0,
                                              .inverseMass = 1.0 / propMass,
                                              .inverseInertia = glm::dmat3(0.0),
                                              .centre = glm::dvec3(-1.0, 0.0, 0.0)});
        manifold.points.push_back(ContactPoint{.position = glm::dvec3(0.0),
                                               .normal = glm::dvec3(1.0, 0.0, 0.0),
                                               .penetration = 0.0,
                                               .prop = 0,
                                               .body = 1});

        resolveContacts(state, manifold, material, tick);

        return std::pair{state, manifold};
    };

    SECTION("twenty kilograms takes nearly all of it and the car barely slows")
    {
        const auto [state, manifold] = build(20.0);

        const auto effective = 1.0 / (1.0 / 1400.0 + 1.0 / 20.0);
        const auto expected = 5.0 * effective;

        REQUIRE_THAT(manifold.points.front().normalImpulse, WithinRel(expected, 1e-9));

        // The car keeps 98.6% of its approach speed.
        REQUIRE_THAT(state.linearVelocity.x, WithinRel(-5.0 + expected / 1400.0, 1e-9));
        REQUIRE(std::abs(state.linearVelocity.x) > 4.9);

        // And the bin leaves at very nearly the speed the car arrived at.
        REQUIRE_THAT(manifold.bodies[1].deltaLinear.x, WithinRel(-expected / 20.0, 1e-9));
        REQUIRE(manifold.bodies[1].deltaLinear.x < -4.9);

        // Nothing is left approaching, which is what the constraint was for.
        const auto relative = state.linearVelocity.x - manifold.bodies[1].deltaLinear.x;
        REQUIRE_THAT(relative, WithinAbs(0.0, 1e-9));
    }

    SECTION("four hundred and fifty kilograms is felt")
    {
        const auto [state, manifold] = build(450.0);

        // A quarter of the approach speed gone, against one and a half percent for the bin.
        REQUIRE(state.linearVelocity.x > -3.9);
        REQUIRE(state.linearVelocity.x < -3.7);
    }

    SECTION("momentum along the normal is conserved")
    {
        for (const auto propMass : {20.0, 450.0})
        {
            const auto [state, manifold] = build(propMass);

            const auto car = 1400.0 * (state.linearVelocity.x - (-5.0));
            const auto prop = propMass * manifold.bodies[1].deltaLinear.x;

            // Newton-seconds, against an impulse of order 100 to 1700 of them. The tolerance is a
            // rounding floor and not a fitted one: the two numbers are the same impulse with
            // opposite signs, computed through different inverse masses.
            REQUIRE_THAT(car + prop, WithinAbs(0.0, 1e-9));
        }
    }
}

TEST_CASE("the tangent basis is a function of the normal alone", "[physics][contact]")
{
    const auto normals = std::vector<glm::dvec3>{
        glm::dvec3(0.0, 1.0, 0.0),
        glm::dvec3(1.0, 0.0, 0.0),
        glm::dvec3(0.0, 0.0, -1.0),
        glm::normalize(glm::dvec3(0.82, 0.12, -0.56)),
        glm::normalize(glm::dvec3(1.0, 1.0, 1.0)),
    };

    for (const auto& normal : normals)
    {
        const auto basis = tangentBasis(normal);

        REQUIRE_THAT(glm::length(basis.first), WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(glm::length(basis.second), WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(glm::dot(basis.first, normal), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(glm::dot(basis.second, normal), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(glm::dot(basis.first, basis.second), WithinAbs(0.0, 1e-12));

        // **Bit-identical on a second call**, which is the property the accumulators depend on and
        // the reason the axis is picked by comparison rather than by anything continuous: a basis
        // that drifted between iterations would be adding numbers that mean different things.
        const auto again = tangentBasis(normal);
        for (auto axis = 0; axis < 3; axis++)
        {
            REQUIRE(basis.first[axis] == again.first[axis]);
            REQUIRE(basis.second[axis] == again.second[axis]);
        }
    }
}

TEST_CASE("friction is clamped as a pair against the cone", "[physics][contact]")
{
    const auto material = plainMaterial();

    SECTION("a binding cone delivers exactly mu times the normal impulse, opposing the slide")
    {
        // Pressed at 2 m/s and sliding at 10. Cancelling the slide outright would take 14000 N·s;
        // the cone allows 0.6 x 2800.
        const auto sliding = glm::dvec3(10.0, -2.0, 0.0);

        auto state = pointMass(sliding);
        auto manifold = flatContact();

        resolveContacts(state, manifold, material, tick);

        const auto& point = manifold.points.front();
        const auto friction = point.tangentImpulse1 * point.tangent1 + point.tangentImpulse2 * point.tangent2;

        REQUIRE_THAT(point.normalImpulse, WithinRel(2800.0, 1e-9));
        REQUIRE_THAT(glm::length(friction), WithinRel(material.friction * point.normalImpulse, 1e-9));

        // **And it points the right way**, which is the thing a scalar accumulator could not be
        // asked. The slide is along +x, so the impulse is along −x and nothing else.
        REQUIRE_THAT(glm::dot(glm::normalize(friction), glm::dvec3(-1.0, 0.0, 0.0)), WithinAbs(1.0, 1e-12));

        // Still sliding, because the cone ran out: 10 − 1680/1400.
        REQUIRE_THAT(state.linearVelocity.x, WithinRel(10.0 - 1680.0 / 1400.0, 1e-9));
    }

    SECTION("a diagonal slide is held to the same cone, not to a box around it")
    {
        // The discriminating case, and the one clamping each axis on its own gets wrong. Sliding at
        // 45 degrees to the basis, a box clamp allows sqrt(2) x mu x jn — a friction coefficient 40%
        // higher on this contact than on the one above, decided by which way a face happened to
        // point.
        const auto sliding = glm::normalize(glm::dvec3(1.0, 0.0, 1.0)) * 10.0 + glm::dvec3(0.0, -2.0, 0.0);

        auto state = pointMass(sliding);
        auto manifold = flatContact();

        resolveContacts(state, manifold, material, tick);

        const auto& point = manifold.points.front();
        const auto friction = point.tangentImpulse1 * point.tangent1 + point.tangentImpulse2 * point.tangent2;

        REQUIRE_THAT(glm::length(friction), WithinRel(material.friction * point.normalImpulse, 1e-9));

        const auto direction = glm::normalize(glm::dvec3(sliding.x, 0.0, sliding.z));
        REQUIRE_THAT(glm::dot(glm::normalize(friction), -direction), WithinAbs(1.0, 1e-12));
    }

    SECTION("under the cone it is unclamped and the slide stops")
    {
        // 1 m/s of slide is 1400 N·s of impulse against a 1680 N·s cone, so nothing binds.
        auto state = pointMass(glm::dvec3(1.0, -2.0, 0.0));
        auto manifold = flatContact();

        resolveContacts(state, manifold, material, tick);

        const auto& point = manifold.points.front();
        const auto friction = point.tangentImpulse1 * point.tangent1 + point.tangentImpulse2 * point.tangent2;

        REQUIRE(glm::length(friction) < material.friction * point.normalImpulse);
        REQUIRE_THAT(glm::length(friction), WithinRel(1400.0, 1e-9));

        REQUIRE_THAT(state.linearVelocity.x, WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(state.linearVelocity.y, WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("an anchor is judged on the collision the pair would have, not on the arrest", "[physics][contact]")
{
    // **The fix for the flying bins, stated as arithmetic.**
    //
    // While a prop is bolted down the solver resolves against an immovable body, so the impulse it
    // computes is whatever it takes to stop the car — `m_car x v`, with the prop's own mass nowhere
    // in it. Handing that to a twenty-kilogram bin gave the bin `m_car / m_bin` times the car's
    // approach speed, about seventy times. Scaling it afterwards is not enough either: by then the
    // car's momentum has gone into an anchor that was about to break, and the car is stopped dead by
    // a bin. So the anchor is asked first, from the impulse the *pair* would exchange.
    const auto material = plainMaterial();

    const auto build = [&material](const double propMass, const double breakForce)
    {
        auto state = pointMass(glm::dvec3(-5.0, 0.0, 0.0));
        state.position = glm::dvec3(1.0, 0.0, 0.0);

        auto manifold = ContactManifold{};
        manifold.bodies.push_back(ContactBody{});
        manifold.bodies.push_back(ContactBody{.prop = 0,
                                              .centre = glm::dvec3(-1.0, 0.0, 0.0),
                                              .anchored = true,
                                              .breakForce = breakForce,
                                              .breakTorque = 1.0e12,
                                              .freeInverseMass = 1.0 / propMass});
        manifold.points.push_back(ContactPoint{.position = glm::dvec3(0.0),
                                               .normal = glm::dvec3(1.0, 0.0, 0.0),
                                               .penetration = 0.0,
                                               .prop = 0,
                                               .body = 1});

        releaseBrokenProps(state, manifold, tick);
        resolveContacts(state, manifold, material, tick);

        return std::pair{state, manifold};
    };

    SECTION("a bin whose anchor gives leaves at about the speed the car arrived at")
    {
        // The pair impulse is 5 / (1/1400 + 1/20) = 98.6 N.s, which over a tick is 35.5 kN against a
        // 500 N anchor.
        const auto [state, manifold] = build(20.0, 500.0);

        REQUIRE(manifold.bodies[1].released);
        REQUIRE_FALSE(manifold.bodies[1].anchored);

        const auto expected = 5.0 / (1.0 / 1400.0 + 1.0 / 20.0);
        REQUIRE_THAT(manifold.points.front().normalImpulse, WithinRel(expected, 1e-9));

        // The bin leaves at 4.93 m/s and not at 350. **This is the number that was wrong.**
        REQUIRE_THAT(manifold.bodies[1].deltaLinear.x, WithinRel(-expected / 20.0, 1e-9));
        REQUIRE(std::abs(manifold.bodies[1].deltaLinear.x) < 5.0);

        // And the car keeps 98.6% of its approach speed rather than being stopped.
        REQUIRE(state.linearVelocity.x < -4.9);
    }

    SECTION("a bollard whose anchor holds still stops the car dead")
    {
        const auto [state, manifold] = build(20.0, 1.0e9);

        REQUIRE_FALSE(manifold.bodies[1].released);
        REQUIRE(manifold.bodies[1].anchored);

        // The arrest impulse, and it is 71 times the pair's. Left where it belongs — on something
        // that did not move.
        REQUIRE_THAT(manifold.points.front().normalImpulse, WithinRel(5.0 * 1400.0, 1e-9));
        REQUIRE_THAT(state.linearVelocity.x, WithinAbs(0.0, 1e-9));
        REQUIRE(manifold.bodies[1].deltaLinear == glm::dvec3(0.0));
    }

    SECTION("a heavier prop loads its own anchor harder at the same speed")
    {
        // 450 kg against 20 kg at 5 m/s is 1703 N.s against 98.6 — seventeen times the load on the
        // base, which is the right direction and is not something the arrest impulse could express:
        // that one is `m_car x v` for both.
        const auto light = build(20.0, 100000.0);
        const auto heavy = build(450.0, 100000.0);

        REQUIRE_FALSE(light.second.bodies[1].released);
        REQUIRE(heavy.second.bodies[1].released);
    }

    SECTION("a contact that is coming apart carries no load at all")
    {
        auto state = pointMass(glm::dvec3(5.0, 0.0, 0.0));
        state.position = glm::dvec3(1.0, 0.0, 0.0);

        auto manifold = ContactManifold{};
        manifold.bodies.push_back(ContactBody{});
        manifold.bodies.push_back(ContactBody{.prop = 0,
                                              .centre = glm::dvec3(-1.0, 0.0, 0.0),
                                              .anchored = true,
                                              .breakForce = 1.0,
                                              .breakTorque = 1.0,
                                              .freeInverseMass = 1.0 / 20.0});
        manifold.points.push_back(ContactPoint{.position = glm::dvec3(0.0),
                                               .normal = glm::dvec3(1.0, 0.0, 0.0),
                                               .penetration = 0.0,
                                               .prop = 0,
                                               .body = 1});

        // Driving away from it, with thresholds a breath would exceed. A separating contact must not
        // load an anchor, or a car reversing off a post would snap it.
        releaseBrokenProps(state, manifold, tick);

        REQUIRE_FALSE(manifold.bodies[1].released);
        REQUIRE(manifold.bodies[1].anchored);
    }

    SECTION("the torque threshold releases a post on its own")
    {
        // A blow near the top of a column is a couple. The same pair impulse three metres above the
        // centre of mass is well under a large force threshold and well over a small moment one.
        auto state = pointMass(glm::dvec3(-5.0, 0.0, 0.0));
        state.position = glm::dvec3(1.0, 3.0, 0.0);

        auto manifold = ContactManifold{};
        manifold.bodies.push_back(ContactBody{});
        manifold.bodies.push_back(ContactBody{.prop = 0,
                                              .centre = glm::dvec3(-1.0, 0.0, 0.0),
                                              .anchored = true,
                                              .breakForce = 1.0e9,
                                              .breakTorque = 1000.0,
                                              .freeInverseMass = 1.0 / 20.0});
        manifold.points.push_back(ContactPoint{.position = glm::dvec3(0.0, 3.0, 0.0),
                                               .normal = glm::dvec3(1.0, 0.0, 0.0),
                                               .penetration = 0.0,
                                               .prop = 0,
                                               .body = 1});

        releaseBrokenProps(state, manifold, tick);

        REQUIRE(manifold.bodies[1].released);
    }
}
