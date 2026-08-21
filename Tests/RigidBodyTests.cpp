#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::angularVelocity;
using raceengine::computeMassProperties;
using raceengine::ForceAccumulator;
using raceengine::integrate;
using raceengine::kineticEnergy;
using raceengine::MassComponent;
using raceengine::RigidBodyState;
using raceengine::setAngularVelocity;

namespace
{

// The vehicle rate the brief fixes, and the engine's own simulation rate, which the vehicle will
// substep three times inside. Both appear here because criterion 1 asks the integrator to be
// stable at the slower one, not merely at the faster.
constexpr auto vehicleStep = 1.0 / 360.0;
constexpr auto engineStep = 1.0 / 120.0;

constexpr auto gravity = 9.80665;

// A body whose principal inertias are all different, which is what the intermediate-axis case
// needs: with two equal there is no unstable axis to find.
RigidBodyState asymmetricBody()
{
    auto state = RigidBodyState{};
    state.mass = 1.0;
    state.inertia = glm::dmat3(1.0);
    state.inertia[0][0] = 1.0;
    state.inertia[1][1] = 2.0;
    state.inertia[2][2] = 3.0;
    state.inverseInertia = glm::inverse(state.inertia);

    return state;
}

} // namespace

TEST_CASE("a body under gravity alone traces the arc its timestep prescribes", "[physics][integrator]")
{
    auto state = RigidBodyState{};
    state.mass = 1200.0;
    state.linearVelocity = glm::dvec3(20.0, 5.0, 0.0);

    const auto ticks = 720; // two seconds at 360 Hz
    for (auto tick = 0; tick < ticks; tick++)
    {
        auto forces = ForceAccumulator{};
        // Gravity as a force and not as a term inside the integrator, so that a body in free fall
        // and a body on its springs are integrated by the same code.
        forces.force = glm::dvec3(0.0, -gravity * state.mass, 0.0);

        integrate(state, forces, vehicleStep);
    }

    const auto elapsed = static_cast<double>(ticks) * vehicleStep;

    // Semi-implicit Euler's exact answer, not the continuum's: position advances by the velocity
    // *after* the acceleration is applied, so the fall is the triangular number of steps rather
    // than the integral. Asserting against this rather than against 0.5·g·t² is what makes the
    // test sensitive to a real regression instead of to the discretisation it is supposed to have.
    const auto steps = static_cast<double>(ticks);
    const auto exactDrop = gravity * vehicleStep * vehicleStep * steps * (steps + 1.0) / 2.0;

    REQUIRE(state.position.x == Catch::Approx(20.0 * elapsed).epsilon(1e-12));
    REQUIRE(state.position.y == Catch::Approx(5.0 * elapsed - exactDrop).epsilon(1e-12));
    REQUIRE(state.position.z == Catch::Approx(0.0).margin(1e-15));

    // Nothing acts across the arc, so the horizontal velocity is untouched to the last bit.
    REQUIRE(state.linearVelocity.x == 20.0);
    REQUIRE(state.linearVelocity.z == 0.0);

    // And that exact answer is the continuum's to within half a step of drift — 2.7 cm over two
    // seconds here — which is the discretisation error the fixed step buys, stated rather than
    // discovered later.
    const auto continuum = 5.0 * elapsed - 0.5 * gravity * elapsed * elapsed;
    REQUIRE(std::abs(state.position.y - continuum) < 0.5 * gravity * vehicleStep * elapsed * 1.001);

    // No torque was applied, so the body must not have acquired any rotation at all.
    REQUIRE(glm::length(state.angularMomentum) == 0.0);
}

TEST_CASE("a body spun about its intermediate axis flips", "[physics][integrator]")
{
    auto state = asymmetricBody();

    // Mostly about y, the intermediate axis, with a small nudge off it. A perfectly aligned spin is
    // an equilibrium — an unstable one, but the arithmetic will sit on it forever.
    setAngularVelocity(state, glm::dvec3(0.02, 10.0, 0.0));

    const auto startingMomentum = glm::length(state.angularMomentum);
    const auto startingEnergy = kineticEnergy(state);

    auto signChanges = 0;
    auto previousSign = 1.0;
    auto worstMomentumDrift = 0.0;
    auto worstEnergyDrift = 0.0;

    const auto ticks = 7200; // twenty seconds
    for (auto tick = 0; tick < ticks; tick++)
    {
        integrate(state, ForceAccumulator{}, vehicleStep);

        // Read omega back in the body frame: in the world frame the flip is mixed into the
        // orientation and does not show as a sign change at all.
        const auto bodyOmega = glm::conjugate(state.orientation) * angularVelocity(state);
        const auto sign = bodyOmega.y < 0.0 ? -1.0 : 1.0;
        if (sign != previousSign)
        {
            signChanges++;
            previousSign = sign;
        }

        worstMomentumDrift =
            std::max(worstMomentumDrift, std::abs(glm::length(state.angularMomentum) - startingMomentum));
        worstEnergyDrift = std::max(worstEnergyDrift, std::abs(kineticEnergy(state) - startingEnergy));
    }

    // The tennis racket effect: the spin axis inverts, repeatedly and periodically.
    REQUIRE(signChanges >= 2);

    // Momentum is the integrated state and nothing acted on it, so it is conserved to the bit —
    // not approximately. That exactness is the reason for storing momentum rather than omega.
    REQUIRE(worstMomentumDrift == 0.0);

    // Energy is derived from the orientation rather than integrated, so it is conserved only as
    // well as the orientation is — and unlike the momentum above it is free to wander. Measured, it
    // sits at about 3e-5 and does not grow with the length of the run; the bound is loose enough
    // not to be a tripwire and tight enough that losing the midpoint evaluation, which costs four
    // orders of magnitude here, fails it immediately.
    REQUIRE(worstEnergyDrift / startingEnergy < 1e-3);
}

TEST_CASE("the rotational integrator converges second order", "[physics][integrator]")
{
    // The property behind the bound in the case above, and the one worth pinning: an error that is
    // merely small can be small for the wrong reason, where an order is a statement about the
    // scheme. Ten times the rate must buy a hundred times the accuracy.
    const auto driftOverOneSecond = [](const double rate)
    {
        auto state = asymmetricBody();
        setAngularVelocity(state, glm::dvec3(0.5, 10.0, 0.5));

        const auto startingEnergy = kineticEnergy(state);
        auto worst = 0.0;

        for (auto tick = 0; tick < static_cast<int>(rate); tick++)
        {
            integrate(state, ForceAccumulator{}, 1.0 / rate);
            worst = std::max(worst, std::abs(kineticEnergy(state) - startingEnergy));
        }

        return worst / startingEnergy;
    };

    const auto coarse = driftOverOneSecond(360.0);
    const auto fine = driftOverOneSecond(3600.0);

    REQUIRE(coarse > 0.0);
    REQUIRE(fine > 0.0);
    REQUIRE(coarse / fine > 50.0);
}

TEST_CASE("a symmetric body holds its rotational energy exactly", "[physics][integrator]")
{
    // The control for the two cases above: with equal principal inertias the tensor is the same in
    // every orientation, so omega never changes and there is nothing for a discretisation to get
    // wrong. Anything but machine precision here is a defect in the transforms rather than in the
    // scheme, which is what makes this the first place to look when the tumble cases move.
    auto state = RigidBodyState{};
    state.mass = 1.0;
    state.inertia = glm::dmat3(2.0);
    state.inverseInertia = glm::inverse(state.inertia);
    setAngularVelocity(state, glm::dvec3(0.5, 10.0, 0.5));

    const auto startingEnergy = kineticEnergy(state);
    auto worst = 0.0;

    for (auto tick = 0; tick < 7200; tick++)
    {
        integrate(state, ForceAccumulator{}, vehicleStep);
        worst = std::max(worst, std::abs(kineticEnergy(state) - startingEnergy));
    }

    REQUIRE(worst / startingEnergy < 1e-14);
}

TEST_CASE("a stiff spring-damper settles rather than diverging", "[physics][integrator][damping]")
{
    // A corner of a car on its spring, with a damper an order of magnitude past critical. Nothing
    // on a real car is damped this hard; the point is that the integrator must not care.
    constexpr auto cornerMass = 300.0;
    constexpr auto springRate = 250000.0; // N/m
    constexpr auto damping = 200000.0;    // N·s/m, about 11.5x critical
    constexpr auto startingDisplacement = 0.05;

    const auto settle = [](const double step, const int ticks)
    {
        auto state = RigidBodyState{};
        state.mass = cornerMass;
        state.position = glm::dvec3(0.0, startingDisplacement, 0.0);

        auto worst = 0.0;
        auto crossings = 0;
        auto previousSign = 1.0;

        for (auto tick = 0; tick < ticks; tick++)
        {
            auto forces = ForceAccumulator{};
            forces.force = glm::dvec3(0.0, -springRate * state.position.y, 0.0);
            // The damper as a coefficient rather than as -c·v. This is the line the test exists for.
            forces.linearDamping = glm::dmat3(damping);

            integrate(state, forces, step);

            worst = std::max(worst, std::abs(state.position.y));
            const auto sign = state.position.y < 0.0 ? -1.0 : 1.0;
            if (sign != previousSign)
            {
                crossings++;
                previousSign = sign;
            }
        }

        return std::tuple{state.position.y, worst, crossings};
    };

    SECTION("at 360 Hz it settles without oscillating")
    {
        const auto [final, worst, crossings] = settle(vehicleStep, 3600);

        // Four orders of magnitude down from where it started, and not further: a damper at eleven
        // times critical has a slow root at roughly k/c, so this settles in about a second and a
        // half of simulated time and then creeps. Asking for zero would be asking the overdamping
        // not to be overdamping.
        REQUIRE(std::abs(final) < startingDisplacement * 1e-4);
        // Overdamped: it never overshoots, so it never crosses zero and never exceeds where it began.
        REQUIRE(crossings == 0);
        REQUIRE(worst <= startingDisplacement);
    }

    SECTION("at 120 Hz it does not diverge")
    {
        const auto [final, worst, crossings] = settle(engineStep, 1200);

        // Four orders of magnitude down from where it started, and not further: a damper at eleven
        // times critical has a slow root at roughly k/c, so this settles in about a second and a
        // half of simulated time and then creeps. Asking for zero would be asking the overdamping
        // not to be overdamping.
        REQUIRE(std::abs(final) < startingDisplacement * 1e-4);
        REQUIRE(crossings == 0);
        REQUIRE(worst <= startingDisplacement);
    }

    SECTION("and the same damper applied explicitly would have diverged")
    {
        // Not a test of the integrator — a test that the two above have teeth. Contributing the
        // damper as a force computed from the tick's starting velocity is the obvious way to write
        // this, and at 120 Hz c·dt/m is 5.6, so each correction overshoots by more than it fixed.
        auto position = startingDisplacement;
        auto velocity = 0.0;

        for (auto tick = 0; tick < 60; tick++)
        {
            const auto force = -springRate * position - damping * velocity;
            velocity += (force / cornerMass) * engineStep;
            position += velocity * engineStep;
        }

        REQUIRE(std::abs(position) > 1e6);
    }
}

TEST_CASE("an undamped orbit does not gain energy over ten thousand ticks", "[physics][integrator]")
{
    // A circular Kepler orbit, GM = 1, r = 1, v = 1, period 2*pi. Ten thousand ticks at the vehicle
    // rate is about four and a half orbits.
    auto state = RigidBodyState{};
    state.mass = 1.0;
    state.position = glm::dvec3(1.0, 0.0, 0.0);
    state.linearVelocity = glm::dvec3(0.0, 1.0, 0.0);

    const auto energyOf = [](const RigidBodyState& body)
    {
        const auto radius = glm::length(body.position);
        return 0.5 * glm::dot(body.linearVelocity, body.linearVelocity) - 1.0 / radius;
    };

    const auto startingEnergy = energyOf(state);
    auto samples = std::vector<double>{};
    samples.reserve(10000);

    for (auto tick = 0; tick < 10000; tick++)
    {
        auto forces = ForceAccumulator{};
        const auto radius = glm::length(state.position);
        forces.force = -state.position / (radius * radius * radius);

        integrate(state, forces, vehicleStep);
        samples.push_back(energyOf(state));
    }

    // Bounded, which is what a symplectic integrator buys and an explicit one does not.
    for (const auto energy : samples)
    {
        REQUIRE(std::abs(energy - startingEnergy) / std::abs(startingEnergy) < 0.01);
    }

    // And bounded without a trend: an integrator leaking energy would still pass the bound above
    // for a while, so compare the two ends rather than the extremes.
    const auto mean = [&samples](const size_t from, const size_t to)
    {
        auto total = 0.0;
        for (auto index = from; index < to; index++)
        {
            total += samples[index];
        }

        return total / static_cast<double>(to - from);
    };

    const auto opening = mean(0, 1000);
    const auto closing = mean(9000, 10000);
    REQUIRE(std::abs(closing - opening) / std::abs(startingEnergy) < 1e-4);
}

TEST_CASE("mass properties come off the ledger rather than out of a constant", "[physics][mass]")
{
    SECTION("two equal masses put the centre of gravity between them")
    {
        // Each carries a little inertia of its own, which is not decoration: two *point* masses on
        // an axis have none about that axis, so the tensor is singular and the ledger is rightly
        // refused — see the last section.
        const auto components = std::array{
            MassComponent{.mass = 100.0, .centre = glm::dvec3(-1.0, 0.0, 0.0), .inertia = glm::dmat3(5.0)},
            MassComponent{.mass = 100.0, .centre = glm::dvec3(1.0, 0.0, 0.0), .inertia = glm::dmat3(5.0)},
        };

        const auto properties = computeMassProperties(components);
        REQUIRE(properties.has_value());
        REQUIRE(properties->mass == Catch::Approx(200.0));
        REQUIRE(properties->centreOfMass.x == Catch::Approx(0.0).margin(1e-15));

        // Parallel axis: 2 * m * d^2 about the two axes they are offset from, and only their own
        // about the one they lie on.
        REQUIRE(properties->inertia[0][0] == Catch::Approx(10.0));
        REQUIRE(properties->inertia[1][1] == Catch::Approx(210.0));
        REQUIRE(properties->inertia[2][2] == Catch::Approx(210.0));
    }

    SECTION("burning fuel off moves the centre of gravity and lightens the car")
    {
        // The seam the brief asks for, exercised: same chassis, less fuel, and every derived
        // quantity follows without anything being reauthored.
        const auto chassis =
            MassComponent{.mass = 900.0, .centre = glm::dvec3(0.0, 0.30, 0.0), .inertia = glm::dmat3(1200.0)};

        const auto full = std::array{chassis, MassComponent{.mass = 60.0, .centre = glm::dvec3(0.0, 0.25, -1.20)}};
        const auto empty = std::array{chassis, MassComponent{.mass = 5.0, .centre = glm::dvec3(0.0, 0.25, -1.20)}};

        const auto heavy = computeMassProperties(full);
        const auto light = computeMassProperties(empty);
        REQUIRE(heavy.has_value());
        REQUIRE(light.has_value());

        REQUIRE(light->mass < heavy->mass);
        // The tank is behind the centre of gravity, so emptying it moves the centre forward.
        REQUIRE(light->centreOfMass.z > heavy->centreOfMass.z);
        // And a lighter car with its mass gathered closer in resists yaw less.
        REQUIRE(light->inertia[1][1] < heavy->inertia[1][1]);
    }

    SECTION("a ledger that cannot describe a body is reported rather than defaulted")
    {
        REQUIRE_FALSE(computeMassProperties({}).has_value());

        const auto weightless = std::array{MassComponent{.mass = 0.0, .centre = glm::dvec3(0.0)}};
        REQUIRE_FALSE(computeMassProperties(weightless).has_value());

        // One point mass has no extent, so no tensor to invert.
        const auto point = std::array{MassComponent{.mass = 100.0, .centre = glm::dvec3(0.0)}};
        const auto reported = computeMassProperties(point);
        REQUIRE_FALSE(reported.has_value());
        REQUIRE(reported.error().find("concentrated at a point") != std::string::npos);

        // And point masses strung along one axis have none about that axis, which is the case
        // likeliest to reach here from a real vehicle file: an axle written as a row of point
        // masses, or a fuel tank modelled as a line of cells. It is a singular tensor and every
        // torque about that axis would divide by zero, so it is refused at the point the data is
        // read rather than at the first tick that uses it.
        const auto collinear = std::array{
            MassComponent{.mass = 50.0, .centre = glm::dvec3(0.0, 0.0, -1.0)},
            MassComponent{.mass = 50.0, .centre = glm::dvec3(0.0, 0.0, 0.0)},
            MassComponent{.mass = 50.0, .centre = glm::dvec3(0.0, 0.0, 1.0)},
        };
        REQUIRE_FALSE(computeMassProperties(collinear).has_value());
    }
}

TEST_CASE("state round-trips through its own bytes", "[physics][integrator][serialisation]")
{
    // Constraint 2: the validation harness saves and restores this, and rollback will later. A
    // restored body must not merely be close, it must continue identically.
    auto state = asymmetricBody();
    state.linearVelocity = glm::dvec3(31.0, -0.4, 2.0);
    setAngularVelocity(state, glm::dvec3(0.3, 2.0, -0.1));

    auto bytes = std::array<unsigned char, sizeof(RigidBodyState)>{};
    std::memcpy(bytes.data(), &state, sizeof(RigidBodyState));

    auto restored = RigidBodyState{};
    std::memcpy(&restored, bytes.data(), sizeof(RigidBodyState));

    const auto drive = [](RigidBodyState& body)
    {
        for (auto tick = 0; tick < 1000; tick++)
        {
            auto forces = ForceAccumulator{};
            forces.force = glm::dvec3(0.0, -gravity * body.mass, 0.0);
            forces.torque = glm::dvec3(0.05, 0.0, -0.02);
            integrate(body, forces, vehicleStep);
        }
    };

    drive(state);
    drive(restored);

    REQUIRE(std::memcmp(&state, &restored, sizeof(RigidBodyState)) == 0);
}

TEST_CASE("the same inputs produce bit-identical state", "[physics][integrator][determinism]")
{
    // Criterion 12, at the level it has to hold first. Nothing below reads a clock, a global or an
    // unseeded random number, and this is what says so.
    const auto run = []
    {
        auto state = asymmetricBody();
        state.mass = 1150.0;
        setAngularVelocity(state, glm::dvec3(0.1, 0.5, 0.0));

        for (auto tick = 0; tick < 5000; tick++)
        {
            auto forces = ForceAccumulator{};
            const auto phase = static_cast<double>(tick) * vehicleStep;
            forces.force = glm::dvec3(std::sin(phase) * 400.0, -gravity * state.mass, std::cos(phase) * 250.0);
            forces.torque = glm::dvec3(0.0, std::sin(phase * 3.0) * 90.0, 0.0);
            forces.linearDamping = glm::dmat3(35.0);

            integrate(state, forces, vehicleStep);
        }

        return state;
    };

    const auto first = run();
    const auto second = run();

    REQUIRE(std::memcmp(&first, &second, sizeof(RigidBodyState)) == 0);
}
