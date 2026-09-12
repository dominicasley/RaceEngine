// The chassis's side of the unsprung mass's acceleration (docs/chassis-kinematic-reaction-brief.md,
// 2026-09-08 latest of all (d)): the proofs.
//
// The chassis body carries every unsprung mass rigidly at its design wheel centre. The true mass is at
// `x_u = x_B + R·(C(q) − c)` with `ẋ_u = v_B + ω × r + R·C'·q̇`, so the whole car's momentum is
// `P = M·v_B + ω × Σm·Δr + Σm·R·C'·q̇` and its angular momentum about the origin carries
// `Σm·(x_u × ẋ_u − x_d × ẋ_d)` beyond the rigid body's. Their rates are what the chassis must be handed:
//
//     F = −m·[ α × Δr + ω × (ω × Δr) + 2·ω × R·C'·q̇ + R·C''·q̇² ]   at the wheel centre, and
//     τ = −m·Δr × (a_d − g)                                             a couple,
//
// beside the `−m·R·C'·q̈` the force pass always applied. Until this change it applied that alone: a
// wheel swinging on its arc pulled the body with nothing, a wheel sliding on a turning body turned it
// with nothing, and a car alone in the air did not keep its momentum.
//
// Two levels of proof. The element proofs run a one-corner two-body system with closed-form mappings
// — an arc about the sprung mass's centre (`C''·q̇²` alone, `ω = 0`), a uniformly parametrised line
// on a turning body (Coriolis alone, `C'' = 0`), a stretched line (`C'' ∥ C'`, the net reaction
// exactly zero) — through the production scheme (`solveCornerRate`, the reduced inertia,
// `unsprungKinematicReaction`, `integrate`) against a fourth-order integration of the exact seven-
// unknown Newton–Euler system at a step a hundred times finer. The whole-car proofs run `stepVehicle`
// on the Golf alone in the air and keep the true momentum, angular momentum about a fixed origin and
// energy, and Newton for each wheel from its reconstructed world acceleration. Every table prints;
// `./EngineTests "[kinematic-reaction]" -s` shows them.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <expected>
#include <functional>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::angularVelocity;
using raceengine::attachmentAcceleration;
using raceengine::attachmentMobility;
using raceengine::bodyToWorld;
using raceengine::bringUpJolt;
using raceengine::ChassisMotion;
using raceengine::computeMassProperties;
using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::cornerInertiaSlope;
using raceengine::cornerJacobianCurvature;
using raceengine::CornerRateStep;
using raceengine::ForceAccumulator;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::inertiaSlopeStep;
using raceengine::integrate;
using raceengine::KinematicReaction;
using raceengine::linearDamper;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::RigidBodyState;
using raceengine::setAngularVelocity;
using raceengine::solveCornerRate;
using raceengine::solveCornerWithJacobian;
using raceengine::springFreeLengthForLoad;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::UnsprungKinematics;
using raceengine::unsprungKinematicReaction;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::worldInverseInertia;

namespace
{

constexpr auto gravity = 9.80665;
constexpr auto tick = 1.0 / 360.0;

// --- a one-corner two-body system with a closed-form mapping ------------------------------------

struct Mapping
{
    const char* name = "";
    std::function<glm::dvec3(double)> position;  // C(q) − c_s: the mass's position from the SPRUNG centre, body frame
    std::function<glm::dvec3(double)> jacobian;  // C'
    std::function<glm::dvec3(double)> curvature; // C''
};

// An arc of radius `radius` about the sprung centre, in the x–y plane: `C'' ⊥ C'`, `|C'|` constant.
[[nodiscard]] Mapping arc(const double radius, const double sign = 1.0)
{
    return Mapping{.name = sign > 0.0 ? "arc" : "mirrored arc",
                   .position = [=](const double q) { return radius * glm::dvec3(std::sin(q), -sign * std::cos(q), 0.0); },
                   .jacobian = [=](const double q) { return radius * glm::dvec3(std::cos(q), sign * std::sin(q), 0.0); },
                   .curvature = [=](const double q) { return radius * glm::dvec3(-std::sin(q), sign * std::cos(q), 0.0); }};
}

// A uniformly parametrised line through the sprung centre: `C'' = 0`.
[[nodiscard]] Mapping uniformLine()
{
    const auto e = glm::normalize(glm::dvec3(1.0, 0.0, 0.0));
    return Mapping{.name = "uniform line",
                   .position = [=](const double q) { return q * e; },
                   .jacobian = [=](const double) { return e; },
                   .curvature = [=](const double) { return glm::dvec3(0.0); }};
}

// A stretched line through the sprung centre: `C'' ∥ C'`, so the coordinate's own term and the chassis's
// centripetal term are one force with opposite signs and the net reaction is exactly nothing.
[[nodiscard]] Mapping stretchedLine(const double a)
{
    const auto e = glm::normalize(glm::dvec3(0.0, 1.0, 0.0));
    return Mapping{.name = "stretched line",
                   .position = [=](const double q) { return (q + a * q * q) * e; },
                   .jacobian = [=](const double q) { return (1.0 + 2.0 * a * q) * e; },
                   .curvature = [=](const double) { return 2.0 * a * e; }};
}

struct TwoBody
{
    double sprungMass = 1200.0;
    glm::dmat3 sprungInertia{1.0}; // about the sprung centre, body axes
    double unsprungMass = 50.0;
    Mapping mapping;
    double designAngle = 0.0; // where the mass sits in the body's ledger
};

[[nodiscard]] TwoBody golfLike(const Mapping& mapping)
{
    auto body = TwoBody{.mapping = mapping};
    body.sprungInertia = glm::dmat3(1.0);
    body.sprungInertia[0][0] = 2000.0;
    body.sprungInertia[1][1] = 2200.0;
    body.sprungInertia[2][2] = 450.0;
    return body;
}

// The state of the two-body system as the engine keeps it: one rigid body of the whole mass with the
// unsprung mass at its design point, and the corner's coordinate.
struct TwoBodyState
{
    RigidBodyState chassis;
    double q = 0.0;
    double rate = 0.0;
};

// The design ledger: the body's centre `c` from the sprung centre, and its tensor about `c`.
struct Ledger
{
    glm::dvec3 centre{0.0}; // c − c_s, body frame
    glm::dvec3 design{0.0}; // C(q_d) − c_s
    double mass = 0.0;
    glm::dmat3 inertia{0.0};
};

[[nodiscard]] Ledger ledgerOf(const TwoBody& body)
{
    auto ledger = Ledger{};
    ledger.design = body.mapping.position(body.designAngle);
    ledger.mass = body.sprungMass + body.unsprungMass;
    ledger.centre = body.unsprungMass * ledger.design / ledger.mass;
    const auto skew = [](const glm::dvec3& r)
    {
        return glm::dot(r, r) * glm::dmat3(1.0) - glm::outerProduct(r, r);
    };
    ledger.inertia = body.sprungInertia + body.sprungMass * skew(-ledger.centre) +
                     body.unsprungMass * skew(ledger.design - ledger.centre);
    return ledger;
}

[[nodiscard]] TwoBodyState primed(const TwoBody& body, const glm::dvec3& velocity, const glm::dvec3& omega,
                                  const double q, const double rate)
{
    const auto ledger = ledgerOf(body);
    auto state = TwoBodyState{};
    state.chassis.mass = ledger.mass;
    state.chassis.centreOfMass = ledger.centre; // relative to the sprung centre as the model origin
    state.chassis.inertia = ledger.inertia;
    state.chassis.inverseInertia = glm::inverse(ledger.inertia);
    state.chassis.position = glm::dvec3(0.0);
    state.chassis.linearVelocity = velocity;
    setAngularVelocity(state.chassis, omega);
    state.q = q;
    state.rate = rate;
    return state;
}

// The true world-space quantities of the two-body system from the engine's state.
struct World
{
    glm::dvec3 sprungPosition{0.0};
    glm::dvec3 sprungVelocity{0.0};
    glm::dvec3 massPosition{0.0};
    glm::dvec3 massVelocity{0.0};
    glm::dvec3 momentum{0.0};
    glm::dvec3 angularMomentum{0.0}; // about the fixed origin
    double energy = 0.0;
};

[[nodiscard]] World worldOf(const TwoBody& body, const TwoBodyState& state)
{
    const auto& chassis = state.chassis;
    const auto omega = angularVelocity(chassis);
    auto world = World{};
    world.sprungPosition = bodyToWorld(chassis, glm::dvec3(0.0));
    world.sprungVelocity = chassis.linearVelocity + glm::cross(omega, world.sprungPosition - chassis.position);
    world.massPosition = bodyToWorld(chassis, body.mapping.position(state.q));
    world.massVelocity = chassis.linearVelocity + glm::cross(omega, world.massPosition - chassis.position) +
                         chassis.orientation * body.mapping.jacobian(state.q) * state.rate;
    world.momentum = body.sprungMass * world.sprungVelocity + body.unsprungMass * world.massVelocity;
    const auto sprungInertiaWorld =
        glm::mat3_cast(chassis.orientation) * body.sprungInertia * glm::transpose(glm::mat3_cast(chassis.orientation));
    world.angularMomentum = sprungInertiaWorld * omega +
                            body.sprungMass * glm::cross(world.sprungPosition, world.sprungVelocity) +
                            body.unsprungMass * glm::cross(world.massPosition, world.massVelocity);
    world.energy = 0.5 * body.sprungMass * glm::dot(world.sprungVelocity, world.sprungVelocity) +
                   0.5 * glm::dot(omega, sprungInertiaWorld * omega) +
                   0.5 * body.unsprungMass * glm::dot(world.massVelocity, world.massVelocity);
    return world;
}

enum class Scheme
{
    Old,      // the chassis receives −m·R·C'·q̈ alone
    Corrected // and the kinematic reaction
};

struct Tick
{
    glm::dvec3 reaction{0.0};   // what the chassis was handed at the wheel centre, N
    glm::dvec3 couple{0.0};     // and as a couple, N·m
    glm::dvec3 chassisAcceleration{0.0};
    glm::dvec3 massAcceleration{0.0}; // reconstructed from the velocity change
    double rate = 0.0;
};

// One tick of the production scheme on the two-body system, no gravity and no forces: the corner's
// frame and geometry terms with its own reaction through the reduced inertia, the kinematic reaction
// solved to consistency by the same fixed point `stepVehicle` runs, then the rigid body integrator.
[[nodiscard]] Tick advance(const TwoBody& body, TwoBodyState& state, const Scheme scheme, const double dt)
{
    const auto& mapping = body.mapping;
    const auto ledger = ledgerOf(body);
    auto& chassis = state.chassis;
    const auto m = body.unsprungMass;
    const auto omega = angularVelocity(chassis);
    const auto inverseInertia = worldInverseInertia(chassis);
    const auto orientation = chassis.orientation;

    const auto jacobianBody = mapping.jacobian(state.q);
    const auto jacobian = orientation * jacobianBody;
    const auto curvature = orientation * mapping.curvature(state.q);
    const auto arm = orientation * (mapping.position(state.q) - ledger.centre);
    const auto designArm = orientation * (ledger.design - ledger.centre);
    const auto inertia = m * glm::dot(jacobianBody, jacobianBody);
    const auto slope = 2.0 * m * glm::dot(jacobianBody, mapping.curvature(state.q));

    const auto motion = ChassisMotion{.acceleration = glm::dvec3(0.0),
                                      .angularVelocity = omega,
                                      .angularAcceleration = inverseInertia * (-glm::cross(omega, chassis.angularMomentum))};
    const auto mobility = attachmentMobility(chassis.mass, inverseInertia, arm, jacobian, arm, jacobian);
    const auto reduced = inertia - m * m * mobility;
    const auto linearResponse = jacobian / chassis.mass;
    const auto angularResponse = inverseInertia * glm::cross(arm, jacobian);
    auto wheel = UnsprungKinematics{
        .mass = m, .jacobian = jacobian, .curvature = curvature, .arm = arm, .designArm = designArm, .rate = state.rate};

    auto kinematic = KinematicReaction{};
    auto kinematicLinear = glm::dvec3(0.0);
    auto kinematicAngular = glm::dvec3(0.0);
    auto rateChange = 0.0;

    for (auto sweep = 0; sweep < 64; sweep++)
    {
        auto kinematicMoved = 0.0;
        if (scheme == Scheme::Corrected)
        {
            auto estimate = motion;
            const auto force = -m * rateChange / dt;
            estimate.acceleration += force * linearResponse + kinematicLinear;
            estimate.angularAcceleration += force * angularResponse + kinematicAngular;
            // The solved rate, the secant curvature over the tick's travel and the path couple, as the
            // production sweep reads them.
            wheel.rate = state.rate + rateChange;
            const auto travel = wheel.rate * dt;
            auto pathCouple = glm::dvec3(0.0);
            wheel.curvature = curvature;
            if (std::abs(travel) > 1e-9)
            {
                wheel.curvature = orientation * (mapping.jacobian(state.q + travel) - jacobianBody) / travel;
                const auto departure = orientation * (mapping.position(state.q + travel) - mapping.position(state.q)) - jacobian * travel;
                const auto endVelocity = chassis.linearVelocity + estimate.acceleration * dt +
                                         glm::cross(omega + estimate.angularAcceleration * dt, arm) + jacobian * wheel.rate;
                pathCouple = m * glm::cross(departure, endVelocity) / dt;
            }
            auto endSpin = estimate;
            endSpin.angularVelocity = omega + estimate.angularAcceleration * dt;
            kinematic = unsprungKinematicReaction(wheel, endSpin, glm::dvec3(0.0));
            kinematic.couple += pathCouple;
            const auto linear = kinematic.force / chassis.mass;
            const auto angular = inverseInertia * (glm::cross(arm, kinematic.force) + kinematic.couple);
            kinematicMoved = std::max(glm::length(linear - kinematicLinear), glm::length(angular - kinematicAngular)) * dt;
            kinematicLinear = linear;
            kinematicAngular = angular;
        }

        const auto known = attachmentAcceleration(motion, arm) + kinematicLinear + glm::cross(kinematicAngular, arm);
        const auto generalised = -m * glm::dot(jacobian, known) - 0.5 * slope * state.rate * state.rate;
        const auto rate = solveCornerRate(CornerRateStep{.previousRate = state.rate,
                                                         .generalisedForce = generalised,
                                                         .generalisedInertia = reduced,
                                                         .damperCoefficient = 0.0,
                                                         .stopCoefficient = 0.0,
                                                         .deltaTime = dt});
        const auto change = rate - state.rate;
        const auto moved = std::max(std::abs(change - rateChange), kinematicMoved);
        rateChange = change;
        if (moved <= 1e-13)
        {
            break;
        }
    }

    const auto before = worldOf(body, state);
    auto forces = ForceAccumulator{};
    const auto reaction = -m * (rateChange / dt) * jacobian;
    forces.addForceAtPoint(reaction, chassis.position + arm, chassis.position);
    if (scheme == Scheme::Corrected)
    {
        forces.addForceAtPoint(kinematic.force, chassis.position + arm, chassis.position);
        forces.torque += kinematic.couple;
    }

    const auto velocityBefore = chassis.linearVelocity;
    integrate(chassis, forces, dt);
    state.rate += rateChange;
    state.q += state.rate * dt;
    const auto after = worldOf(body, state);

    return Tick{.reaction = reaction + (scheme == Scheme::Corrected ? kinematic.force : glm::dvec3(0.0)),
                .couple = scheme == Scheme::Corrected ? kinematic.couple : glm::dvec3(0.0),
                .chassisAcceleration = (chassis.linearVelocity - velocityBefore) / dt,
                .massAcceleration = (after.massVelocity - before.massVelocity) / dt,
                .rate = state.rate};
}

// --- the exact reference: the seven-unknown Newton–Euler system, fourth order at a fine step --------

struct Exact
{
    glm::dvec3 position{0.0}; // of the sprung centre
    glm::dvec3 velocity{0.0};
    glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
    glm::dvec3 omega{0.0};
    double q = 0.0;
    double rate = 0.0;
};

struct ExactRate
{
    glm::dvec3 velocity{0.0};
    glm::dvec3 acceleration{0.0};
    glm::dvec3 omega{0.0};
    glm::dvec3 alpha{0.0};
    double rate = 0.0;
    double acceleration_q = 0.0;
};

// Solve the exact equations for (a_s, α, q̈): the sprung body (M_s, J_s) about its own centre and a
// point mass at `x_s + R·C(q)` with `ẍ_u = a_s + α × ρ + ω × (ω × ρ) + 2ω × R·C'·q̇ + R·C'·q̈ + R·C''·q̇²`,
// `ρ = R·C(q)`: linear momentum `M_s·a_s + m·ẍ_u = 0`, angular momentum about the sprung centre
// `J_s·α + ω × J_s·ω + m·ρ × ẍ_u = 0` (the constraint's moment is `ρ × F` and `F = m·ẍ_u`), and the
// coordinate `m·R·C'·ẍ_u = 0` (no force along the mass's own freedom).
[[nodiscard]] ExactRate exactRate(const TwoBody& body, const Exact& state)
{
    const auto& mapping = body.mapping;
    const auto R = glm::mat3_cast(state.orientation);
    const auto b = R * mapping.jacobian(state.q);
    const auto k = R * mapping.curvature(state.q);
    const auto rho = R * mapping.position(state.q);
    const auto Js = R * body.sprungInertia * glm::transpose(R);
    const auto& w = state.omega;
    const auto m = body.unsprungMass;
    const auto Ms = body.sprungMass;
    const auto known = glm::cross(w, glm::cross(w, rho)) + 2.0 * glm::cross(w, b * state.rate) + k * state.rate * state.rate;

    // Unknowns x = (a_s, α, q̈). ẍ_u = a_s − [ρ]×α + b·q̈ + known.
    auto A = std::array<std::array<double, 7>, 7>{};
    auto rhs = std::array<double, 7>{};
    const auto skew = [](const glm::dvec3& r)
    {
        // [r]× as a matrix acting on a column vector v: r × v
        return glm::dmat3(0.0, r.z, -r.y, -r.z, 0.0, r.x, r.y, -r.x, 0.0);
    };
    const auto rhoSkew = skew(rho); // rho × v
    // rows 0..2: M_s a_s + m (a_s − ρ × α + b q̈) = −m known
    for (auto i = 0; i < 3; i++)
    {
        A[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] += Ms + m;
        for (auto j = 0; j < 3; j++)
        {
            A[static_cast<std::size_t>(i)][static_cast<std::size_t>(3 + j)] += -m * rhoSkew[j][i];
        }
        A[static_cast<std::size_t>(i)][6] += m * b[i];
        rhs[static_cast<std::size_t>(i)] = -m * known[i];
    }
    // rows 3..5: J_s α + m ρ × (a_s − ρ × α + b q̈) = −ω × J_s ω − m ρ × known
    const auto rhoRho = rhoSkew * rhoSkew; // ρ × (ρ × v)
    const auto gyro = glm::cross(w, Js * w);
    const auto rhoKnown = glm::cross(rho, known);
    const auto rhoB = glm::cross(rho, b);
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++)
        {
            A[static_cast<std::size_t>(3 + i)][static_cast<std::size_t>(j)] += m * rhoSkew[j][i];
            A[static_cast<std::size_t>(3 + i)][static_cast<std::size_t>(3 + j)] += Js[j][i] - m * rhoRho[j][i];
        }
        A[static_cast<std::size_t>(3 + i)][6] += m * rhoB[i];
        rhs[static_cast<std::size_t>(3 + i)] = -gyro[i] - m * rhoKnown[i];
    }
    // row 6: b·(a_s − ρ × α + b q̈) = −b·known
    for (auto j = 0; j < 3; j++)
    {
        A[6][static_cast<std::size_t>(j)] += b[j];
        const auto column = glm::dvec3(rhoSkew[j][0], rhoSkew[j][1], rhoSkew[j][2]); // ρ × e_j
        A[6][static_cast<std::size_t>(3 + j)] += -glm::dot(b, column);
    }
    A[6][6] += glm::dot(b, b);
    rhs[6] = -glm::dot(b, known);

    // Gaussian elimination with partial pivoting.
    for (auto col = std::size_t{0}; col < 7; col++)
    {
        auto pivot = col;
        for (auto row = col + 1; row < 7; row++)
        {
            if (std::abs(A[row][col]) > std::abs(A[pivot][col]))
            {
                pivot = row;
            }
        }
        std::swap(A[col], A[pivot]);
        std::swap(rhs[col], rhs[pivot]);
        for (auto row = col + 1; row < 7; row++)
        {
            const auto factor = A[row][col] / A[col][col];
            for (auto k2 = col; k2 < 7; k2++)
            {
                A[row][k2] -= factor * A[col][k2];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    auto x = std::array<double, 7>{};
    for (auto row = std::size_t{7}; row-- > 0;)
    {
        auto sum = rhs[row];
        for (auto k2 = row + 1; k2 < 7; k2++)
        {
            sum -= A[row][k2] * x[k2];
        }
        x[row] = sum / A[row][row];
    }

    return ExactRate{.velocity = state.velocity,
                     .acceleration = glm::dvec3(x[0], x[1], x[2]),
                     .omega = state.omega,
                     .alpha = glm::dvec3(x[3], x[4], x[5]),
                     .rate = state.rate,
                     .acceleration_q = x[6]};
}

[[nodiscard]] Exact plus(const Exact& state, const ExactRate& rate, const double h)
{
    auto next = state;
    next.position += rate.velocity * h;
    next.velocity += rate.acceleration * h;
    const auto speed = glm::length(rate.omega);
    if (speed > 0.0)
    {
        const auto half = 0.5 * speed * h;
        next.orientation = glm::normalize(glm::dquat(std::cos(half), (rate.omega / speed) * std::sin(half)) * state.orientation);
    }
    next.omega += rate.alpha * h;
    next.q += rate.rate * h;
    next.rate += rate.acceleration_q * h;
    return next;
}

void rungeKutta(const TwoBody& body, Exact& state, const double h)
{
    const auto k1 = exactRate(body, state);
    const auto k2 = exactRate(body, plus(state, k1, 0.5 * h));
    const auto k3 = exactRate(body, plus(state, k2, 0.5 * h));
    const auto k4 = exactRate(body, plus(state, k3, h));
    auto blend = ExactRate{};
    blend.velocity = (k1.velocity + 2.0 * k2.velocity + 2.0 * k3.velocity + k4.velocity) / 6.0;
    blend.acceleration = (k1.acceleration + 2.0 * k2.acceleration + 2.0 * k3.acceleration + k4.acceleration) / 6.0;
    blend.omega = (k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega) / 6.0;
    blend.alpha = (k1.alpha + 2.0 * k2.alpha + 2.0 * k3.alpha + k4.alpha) / 6.0;
    blend.rate = (k1.rate + 2.0 * k2.rate + 2.0 * k3.rate + k4.rate) / 6.0;
    blend.acceleration_q = (k1.acceleration_q + 2.0 * k2.acceleration_q + 2.0 * k3.acceleration_q + k4.acceleration_q) / 6.0;
    state = plus(state, blend, h);
}

// The exact state's world quantities, for comparison with the engine's.
[[nodiscard]] World worldOfExact(const TwoBody& body, const Exact& state)
{
    const auto R = glm::mat3_cast(state.orientation);
    auto world = World{};
    world.sprungPosition = state.position;
    world.sprungVelocity = state.velocity;
    const auto rho = R * body.mapping.position(state.q);
    world.massPosition = state.position + rho;
    world.massVelocity = state.velocity + glm::cross(state.omega, rho) + R * body.mapping.jacobian(state.q) * state.rate;
    world.momentum = body.sprungMass * world.sprungVelocity + body.unsprungMass * world.massVelocity;
    const auto Js = R * body.sprungInertia * glm::transpose(R);
    world.angularMomentum = Js * state.omega + body.sprungMass * glm::cross(world.sprungPosition, world.sprungVelocity) +
                            body.unsprungMass * glm::cross(world.massPosition, world.massVelocity);
    world.energy = 0.5 * body.sprungMass * glm::dot(state.velocity, state.velocity) + 0.5 * glm::dot(state.omega, Js * state.omega) +
                   0.5 * body.unsprungMass * glm::dot(world.massVelocity, world.massVelocity);
    return world;
}

// The engine's state as an exact state, for the same start.
[[nodiscard]] Exact exactOf(const TwoBody& body, const TwoBodyState& state)
{
    const auto world = worldOf(body, state);
    return Exact{.position = world.sprungPosition,
                 .velocity = world.sprungVelocity,
                 .orientation = state.chassis.orientation,
                 .omega = angularVelocity(state.chassis),
                 .q = state.q,
                 .rate = state.rate};
}

struct RunErrors
{
    double momentum = 0.0;        // max |P − P₀| / |P_scale|
    double angularMomentum = 0.0; // max |H − H₀| / H_scale
    double energy = 0.0;          // max |E − E₀| / E₀
    double massPosition = 0.0;    // max |x_u − x_u,ref|, m
    double sprungPosition = 0.0;  // max |x_s − x_s,ref|, m
    double rate = 0.0;            // |q̇ − q̇_ref| at the end
    glm::dvec3 reactionMax{0.0};  // the largest |component| of the applied reaction
    double coupleMax = 0.0;
    double referenceMomentum = 0.0; // the reference's own |P − P₀| / scale: what the exact integration keeps
    double referenceAngular = 0.0;
    double velocity = 0.0;          // max |ẋ_u − ẋ_u,ref|, m/s
    double energyEnd = 0.0;         // (E − E₀) / E₀ at the end, signed: which way the first-order drift goes
};

[[nodiscard]] RunErrors run(const TwoBody& body, const TwoBodyState& start, const Scheme scheme, const double dt,
                            const double seconds, const double momentumScale, const double angularScale)
{
    auto state = start;
    auto exact = exactOf(body, start);
    const auto initial = worldOf(body, start);
    auto errors = RunErrors{};
    const auto ticks = static_cast<int>(std::lround(seconds / dt));
    for (auto step = 0; step < ticks; step++)
    {
        const auto advanced = advance(body, state, scheme, dt);
        for (auto sub = 0; sub < 100; sub++)
        {
            rungeKutta(body, exact, dt / 100.0);
        }
        const auto world = worldOf(body, state);
        const auto reference = worldOfExact(body, exact);
        errors.momentum = std::max(errors.momentum, glm::length(world.momentum - initial.momentum) / momentumScale);
        errors.angularMomentum =
            std::max(errors.angularMomentum, glm::length(world.angularMomentum - initial.angularMomentum) / angularScale);
        errors.energy = std::max(errors.energy, std::abs(world.energy - initial.energy) / initial.energy);
        errors.energyEnd = (world.energy - initial.energy) / initial.energy;
        errors.massPosition = std::max(errors.massPosition, glm::length(world.massPosition - reference.massPosition));
        errors.sprungPosition = std::max(errors.sprungPosition, glm::length(world.sprungPosition - reference.sprungPosition));
        errors.velocity = std::max(errors.velocity, glm::length(world.massVelocity - reference.massVelocity));
        errors.referenceMomentum = std::max(errors.referenceMomentum, glm::length(reference.momentum - initial.momentum) / momentumScale);
        errors.referenceAngular = std::max(errors.referenceAngular, glm::length(reference.angularMomentum - initial.angularMomentum) / angularScale);
        errors.rate = std::abs(state.rate - exact.rate);
        for (auto axis = 0; axis < 3; axis++)
        {
            errors.reactionMax[axis] = std::max(errors.reactionMax[axis], std::abs(advanced.reaction[axis]));
        }
        errors.coupleMax = std::max(errors.coupleMax, glm::length(advanced.couple));
    }
    return errors;
}

// --- the whole car ------------------------------------------------------------------------------

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

[[nodiscard]] std::expected<PhysicsWorld, std::string> plate(const double size)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 2.0;
    descriptor.features = {};
    return PhysicsWorld::create(generateProvingGround(descriptor).value());
}

// The Golf alone in the air: no air, springs at zero load at design (or the shipped preload), no
// damper, no seal friction, no bar, stops out of reach — a conservative closed system under gravity.
[[nodiscard]] VehicleSetup isolatedGolf(const bool kinematicReaction, const bool springs)
{
    auto setup = golfGtiMk7().value();
    setup.aero.clear();
    setup.kinematicReaction = kinematicReaction;
    for (auto& corner : setup.corners)
    {
        corner.springFreeLength = springFreeLengthForLoad(corner, 0.0).value();
        if (!springs)
        {
            corner.springRate = 0.0;
        }
        corner.damper = linearDamper(0.0, 0.0);
        corner.damperFriction = 0.0;
        corner.damperFrictionShape = raceengine::Curve{};
        corner.antiRollRate = 0.0;
        corner.bumpStop.gap = 1.0;
        corner.droopStop.gap = 1.0;
    }
    return setup;
}

void primeMassProperties(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world)
{
    const auto position = state.chassis.position;
    const auto velocity = state.chassis.linearVelocity;
    const auto orientation = state.chassis.orientation;
    REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, 1e-12).has_value());
    state.chassis.position = position;
    state.chassis.linearVelocity = velocity;
    state.chassis.orientation = orientation;
    state.chassis.angularMomentum = glm::dvec3(0.0);
    for (auto& corner : state.corners)
    {
        corner.wishboneAngle = 0.0;
        corner.wishboneRate = 0.0;
    }
}

// The true world quantities of the car from the engine's state: the sprung body from the setup's own
// ledger, the four wheels at their wheel centres with their whole velocities.
struct CarWorld
{
    glm::dvec3 momentum{0.0};
    glm::dvec3 angularMomentum{0.0}; // about the fixed origin
    glm::dvec3 centreOfMass{0.0};
    double kinetic = 0.0;
    double potential = 0.0; // the springs
    std::array<glm::dvec3, cornerCount> wheelVelocity{};
};

[[nodiscard]] CarWorld carWorldOf(const VehicleSetup& setup, const VehicleState& state, const VehicleStep& /*step*/,
                                  const glm::dvec3& origin = glm::dvec3(0.0))
{
    const auto sprung = computeMassProperties(setup.sprung).value();
    const auto& chassis = state.chassis;
    const auto omega = angularVelocity(chassis);
    const auto R = glm::mat3_cast(chassis.orientation);
    auto world = CarWorld{};

    const auto sprungPosition = bodyToWorld(chassis, sprung.centreOfMass);
    const auto sprungVelocity = chassis.linearVelocity + glm::cross(omega, sprungPosition - chassis.position);
    const auto sprungInertiaWorld = R * sprung.inertia * glm::transpose(R);
    world.momentum = sprung.mass * sprungVelocity;
    world.angularMomentum = sprungInertiaWorld * omega + sprung.mass * glm::cross(sprungPosition - origin, sprungVelocity);
    world.centreOfMass = sprung.mass * sprungPosition;
    world.kinetic = 0.5 * sprung.mass * glm::dot(sprungVelocity, sprungVelocity) + 0.5 * glm::dot(omega, sprungInertiaWorld * omega);
    auto totalMass = sprung.mass;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        // The geometry at the state's own angle, not the step's, which was solved a tick earlier.
        const auto suspension = solveCornerWithJacobian(corner.hardpoints, state.corners[index].wishboneAngle, 0.0).value();
        const auto position = bodyToWorld(chassis, suspension.wheelCentre);
        const auto velocity = chassis.linearVelocity + glm::cross(omega, position - chassis.position) +
                              chassis.orientation * suspension.wheelCentrePerAngle * state.corners[index].wishboneRate;
        world.wheelVelocity[index] = velocity;
        world.momentum += corner.unsprungMass * velocity;
        world.angularMomentum += corner.unsprungMass * glm::cross(position - origin, velocity);
        world.centreOfMass += corner.unsprungMass * position;
        world.kinetic += 0.5 * corner.unsprungMass * glm::dot(velocity, velocity);
        totalMass += corner.unsprungMass;

        const auto spring = raceengine::solveElement(corner.hardpoints, raceengine::springElementOf(corner.hardpoints),
                                                     state.corners[index].wishboneAngle);
        const auto stretch = corner.springFreeLength - spring.length;
        world.potential += 0.5 * corner.springRate * stretch * stretch;
    }
    world.centreOfMass /= totalMass;
    return world;
}

} // namespace

// =================================================================================================
// The reaction itself
// =================================================================================================

TEST_CASE("the kinematic reaction is zero where nothing moves, even in the rate where the rate is squared, odd where it is not, and the Golf's curvature is the Jacobian's own",
          "[physics][suspension][kinematic-reaction]")
{
    const auto b = glm::dvec3(0.05, 0.36, -0.02);
    const auto k = glm::dvec3(0.02, -0.03, 0.30);
    const auto arm = glm::dvec3(-0.75, -0.3, 1.3);
    const auto design = glm::dvec3(-0.75, -0.25, 1.3);
    const auto g = glm::dvec3(0.0, -gravity, 0.0);

    // At rest on a level road: no rate, no rotation, the chassis at rest — nothing.
    const auto still = unsprungKinematicReaction(UnsprungKinematics{.mass = 43.0, .jacobian = b, .curvature = k, .arm = arm, .designArm = design, .rate = 0.0},
                                                 ChassisMotion{.acceleration = glm::dvec3(0.0), .angularVelocity = glm::dvec3(0.0), .angularAcceleration = glm::dvec3(0.0)}, glm::dvec3(0.0));
    REQUIRE(still.force == glm::dvec3(0.0));
    REQUIRE(still.couple == glm::dvec3(0.0));

    // In free fall with the wheel displaced: the couple reads the design point against g and is nothing.
    const auto falling = unsprungKinematicReaction(UnsprungKinematics{.mass = 43.0, .jacobian = b, .curvature = k, .arm = arm, .designArm = design, .rate = 0.0},
                                                   ChassisMotion{.acceleration = g, .angularVelocity = glm::dvec3(0.0), .angularAcceleration = glm::dvec3(0.0)}, g);
    REQUIRE(falling.force == glm::dvec3(0.0));
    REQUIRE(glm::length(falling.couple) < 1e-15);

    // The centripetal term is even in the rate; the Coriolis term is odd in it and in ω.
    const auto at = [&](const double rate, const glm::dvec3& omega)
    {
        return unsprungKinematicReaction(UnsprungKinematics{.mass = 43.0, .jacobian = b, .curvature = k, .arm = arm, .designArm = arm, .rate = rate},
                                         ChassisMotion{.acceleration = glm::dvec3(0.0), .angularVelocity = omega, .angularAcceleration = glm::dvec3(0.0)}, glm::dvec3(0.0));
    };
    const auto forward = at(2.0, glm::dvec3(0.0));
    const auto back = at(-2.0, glm::dvec3(0.0));
    REQUIRE(forward.force == back.force);
    REQUIRE(forward.force.x == Catch::Approx(-43.0 * 4.0 * k.x).epsilon(1e-12));
    REQUIRE(forward.force.y == Catch::Approx(-43.0 * 4.0 * k.y).epsilon(1e-12));
    REQUIRE(forward.force.z == Catch::Approx(-43.0 * 4.0 * k.z).epsilon(1e-12));
    const auto omega = glm::dvec3(0.0, 0.0, 1.5);
    const auto coriolis = at(2.0, omega).force - forward.force;
    const auto coriolisBack = at(-2.0, omega).force - back.force;
    const auto coriolisSpin = at(2.0, -omega).force - forward.force;
    REQUIRE(glm::length(coriolis + coriolisBack) < 1e-9);
    REQUIRE(glm::length(coriolis + coriolisSpin) < 1e-9);
    REQUIRE(glm::length(coriolis - (-43.0 * 2.0 * glm::cross(omega, b * 2.0))) < 1e-9);
    std::printf("\n=== the reaction's symmetries: centripetal even (%.3f = %.3f N), Coriolis odd in q̇ and ω (%.3f / %.3f N) ===\n",
                forward.force.x, back.force.x, coriolis.x, coriolisBack.x);

    // The Golf's curvature from the production helper against an independent difference of the solve's Jacobian.
    const auto golf = golfGtiMk7().value();
    for (auto index = std::size_t{0}; index < cornerCount; index += 2)
    {
        const auto& corner = golf.corners[index];
        auto worst = 0.0;
        for (auto sample = 0; sample <= 10; sample++)
        {
            const auto q = corner.hardpoints.droopAngle + (corner.hardpoints.bumpAngle - corner.hardpoints.droopAngle) * sample / 10.0;
            const auto helper = cornerJacobianCurvature(corner, q, 0.0, true);
            const auto h = 5e-4;
            const auto ahead = solveCornerWithJacobian(corner.hardpoints, q + h, 0.0).value().wheelCentrePerAngle;
            const auto behind = solveCornerWithJacobian(corner.hardpoints, q - h, 0.0).value().wheelCentrePerAngle;
            const auto independent = (ahead - behind) / (2.0 * h);
            worst = std::max(worst, glm::length(helper - independent));
            // And the identity that ties it to the shipped slope: I' = 2·m·C'·C''.
            const auto here = solveCornerWithJacobian(corner.hardpoints, q, 0.0).value().wheelCentrePerAngle;
            REQUIRE(2.0 * corner.unsprungMass * glm::dot(here, helper) == Catch::Approx(cornerInertiaSlope(corner, q, 0.0, true)).margin(1e-5));
            if (sample == 5)
            {
                std::printf("  Golf %s at design: C' (%.4f, %.4f, %.4f) m/rad, C'' (%.4f, %.4f, %.4f) m/rad², |C''| %.4f, along C' %.4f, across %.4f; m·|C''|·q̇² at 2 rad/s %.1f N, at 4.7 rad/s %.1f N\n",
                            cornerAbbreviation(static_cast<Corner>(index)), here.x, here.y, here.z, helper.x, helper.y, helper.z,
                            glm::length(helper), glm::dot(helper, here) / glm::length(here),
                            glm::length(helper - glm::dot(helper, here) / glm::dot(here, here) * here),
                            corner.unsprungMass * glm::length(helper) * 4.0, corner.unsprungMass * glm::length(helper) * 4.7 * 4.7);
            }
        }
        REQUIRE(worst < 1e-6);
    }
}

// =================================================================================================
// The element proofs: one corner on a closed-form mapping, against the exact system
// =================================================================================================

TEST_CASE("a wheel swinging on an arc about the sprung centre pulls the body round with it: the C''·q̇² reaction, ω = 0",
          "[physics][suspension][kinematic-reaction]")
{
    // The mass on an arc of 0.4 m about the sprung centre, launched at 3 rad/s with the body still:
    // the exact motion is the mass going round at a constant rate while the sprung centre circles the
    // fixed system centre, the body never turning because the constraint's force passes through its
    // centre. The corner term is nothing (`C' ⊥ C''`), and the whole reaction is the chassis's.
    for (const auto sign : {1.0, -1.0})
    {
        for (const auto rate0 : {3.0, -3.0})
        {
            const auto body = golfLike(arc(0.4, sign));
            const auto start = primed(body, glm::dvec3(0.0), glm::dvec3(0.0), 0.0, rate0);
            const auto initial = worldOf(body, start);
            const auto momentumScale = body.unsprungMass * 0.4 * std::abs(rate0);
            const auto angularScale = body.unsprungMass * 0.4 * 0.4 * std::abs(rate0);

            std::printf("\n=== %s, q̇₀ %+.0f rad/s, 2 s at 360 Hz (P₀ %.3f N·s, H₀ %.3f N·m·s, E₀ %.3f J) ===\n", body.mapping.name, rate0,
                        glm::length(initial.momentum), glm::length(initial.angularMomentum), initial.energy);
            for (const auto scheme : {Scheme::Old, Scheme::Corrected})
            {
                const auto errors = run(body, start, scheme, tick, 2.0, momentumScale, angularScale);
                std::printf("  %-9s |ΔP| %.2e, |ΔH| %.2e, |ΔE| %.2e (end %+.2e); |x_u − ref| %.2e m, |x_s − ref| %.2e m, |q̇ − ref| %.2e; reaction max (%.1f, %.1f, %.1f) N, couple %.2e N·m\n",
                            scheme == Scheme::Old ? "OLD" : "CORRECTED", errors.momentum, errors.angularMomentum, errors.energy,
                            errors.energyEnd, errors.massPosition, errors.sprungPosition, errors.rate, errors.reactionMax.x, errors.reactionMax.y,
                            errors.reactionMax.z, errors.coupleMax);
                if (scheme == Scheme::Old)
                {
                    // The build before: the body never moves, the mass's momentum swings round unbalanced.
                    REQUIRE(errors.momentum > 1.0);
                    REQUIRE(errors.reactionMax.x < 1e-9);
                    REQUIRE(errors.reactionMax.y < 1e-9);
                }
                else
                {
                    // The linear impulse is exact to the secant, the angular one to the path couple:
                    // both close to the solve's rounding. The energy is the semi-implicit step's, first
                    // order — a loss of 0.1 % per second at 360 Hz on this pair, halving with the step
                    // (the sweep below) — and the positions follow the exact pair to a millimetre.
                    REQUIRE(errors.momentum < 1e-5);
                    REQUIRE(errors.angularMomentum < 1e-5);
                    REQUIRE(errors.energy < 3e-3);
                    REQUIRE(errors.energyEnd < 0.0);
                    REQUIRE(errors.sprungPosition < 1e-3);
                    REQUIRE(errors.rate < 5e-3);
                    // The pull is m·ρ·q̇² and it is the same either way round.
                    REQUIRE(std::max(errors.reactionMax.x, errors.reactionMax.y) == Catch::Approx(body.unsprungMass * 0.4 * rate0 * rate0).epsilon(2e-2));
                }
            }
        }
    }
}

TEST_CASE("a wheel sliding on a turning body turns it back: the Coriolis reaction, C'' = 0",
          "[physics][suspension][kinematic-reaction]")
{
    // The mass on a uniform line through the sprung centre, the body spinning at ω about the line's
    // normal, the mass sliding out at q̇: angular momentum about the origin is conserved as the
    // mass's lever grows and the body slows — the classic bead on a spinning rod — and the whole of
    // that exchange is the Coriolis reaction `−2m·ω × R·C'·q̇` on the body. Every sign combination.
    for (const auto spin : {1.5, -1.5})
    {
        for (const auto rate0 : {1.0, -1.0})
        {
            const auto body = golfLike(uniformLine());
            const auto omega = glm::dvec3(0.0, 0.0, spin);
            const auto start = primed(body, glm::dvec3(0.0), omega, 0.3, rate0);
            const auto initial = worldOf(body, start);
            const auto momentumScale = body.unsprungMass * std::abs(rate0);
            const auto angularScale = glm::length(initial.angularMomentum);

            std::printf("\n=== bead on a rod, ω %+.1f rad/s, q̇₀ %+.0f rad/s from q 0.3, 0.5 s at 360 Hz (H₀ %.3f N·m·s) ===\n", spin, rate0, angularScale);
            for (const auto scheme : {Scheme::Old, Scheme::Corrected})
            {
                const auto errors = run(body, start, scheme, tick, 0.5, momentumScale, angularScale);
                std::printf("  %-9s |ΔP| %.2e, |ΔH| %.2e, |ΔE| %.2e (end %+.2e); |x_u − ref| %.2e m, |ẋ_u − ref| %.2e m/s, |x_s − ref| %.2e m, |q̇ − ref| %.2e; reaction max (%.1f, %.1f, %.1f) N, couple %.2e N·m; the reference's own |ΔP| %.1e |ΔH| %.1e\n",
                            scheme == Scheme::Old ? "OLD" : "CORRECTED", errors.momentum, errors.angularMomentum, errors.energy,
                            errors.energyEnd, errors.massPosition, errors.velocity, errors.sprungPosition, errors.rate, errors.reactionMax.x, errors.reactionMax.y,
                            errors.reactionMax.z, errors.coupleMax, errors.referenceMomentum, errors.referenceAngular);
                if (scheme == Scheme::Old)
                {
                    REQUIRE(errors.angularMomentum > 5e-3);
                }
                else
                {
                    // The bead flies out exponentially (`q̈ = ω²·q`) and the explicit Coriolis force is a
                    // tick behind it; first order in the step, the sweep below says by how much.
                    REQUIRE(errors.momentum < 0.1);
                    REQUIRE(errors.angularMomentum < 0.3);
                    REQUIRE(errors.energy < 0.03);
                    REQUIRE(errors.massPosition < 5e-3);
                    REQUIRE(errors.angularMomentum < 0.1 * (scheme == Scheme::Old ? 1.0 : 1.0) + 0.3);
                }
            }
        }
    }
}

TEST_CASE("on a stretched line through the sprung centre the corner's term and the chassis's centripetal term are one force: the net reaction is nothing",
          "[physics][suspension][kinematic-reaction]")
{
    // `C'' ∥ C'`: the mass coasts at a constant world velocity along the line, the corner's
    // `−½I'q̇²` slows `q̇` as the metric stretches, and the chassis must feel nothing at all — the
    // build before handed it `−m·R·C'·q̈` and pushed it along the line for no force.
    const auto body = golfLike(stretchedLine(0.5));
    const auto start = primed(body, glm::dvec3(0.0), glm::dvec3(0.0), 0.0, 1.0);
    const auto momentumScale = body.unsprungMass * 1.0;
    std::printf("\n=== stretched line through the sprung centre, q̇₀ 1 rad/s, 1 s at 360 Hz ===\n");
    for (const auto scheme : {Scheme::Old, Scheme::Corrected})
    {
        const auto errors = run(body, start, scheme, tick, 1.0, momentumScale, 1.0);
        std::printf("  %-9s |ΔP| %.2e, |ΔE| %.2e; |x_s − ref| %.2e m; reaction max (%.2e, %.2e, %.2e) N\n",
                    scheme == Scheme::Old ? "OLD" : "CORRECTED", errors.momentum, errors.energy, errors.sprungPosition,
                    errors.reactionMax.x, errors.reactionMax.y, errors.reactionMax.z);
        if (scheme == Scheme::Old)
        {
            REQUIRE(errors.reactionMax.y > 10.0);
            REQUIRE(errors.momentum > 0.1);
        }
        else
        {
            // The coordinate's own term reads the rate the tick starts with and the chassis's the rate
            // it ends with, so what reaches the chassis is `m·C''·(q̇₁² − q̇₀²)`: one tick's change of
            // rate, 0.3 N here against the 52 N the build before pushed with, first order in the
            // step. The momentum error left is the coordinate's own explicit step
            // (docs/nonlinear-geometry-brief.md §4, 1.9e-3 at 360 Hz).
            REQUIRE(glm::length(errors.reactionMax) < 0.5);
            REQUIRE(errors.momentum < 3e-3);
            REQUIRE(errors.sprungPosition < 1e-4);
        }
    }
}

TEST_CASE("where nothing curves, turns or is displaced the chassis reads the old bits", "[physics][suspension][kinematic-reaction]")
{
    // A uniform line, the body still, the mass at its design point coasting: every term of the
    // reaction is exactly zero and the two schemes' states are the same bits at every tick.
    const auto body = golfLike(uniformLine());
    auto old = primed(body, glm::dvec3(0.0, 0.0, 20.0), glm::dvec3(0.0), 0.0, 0.0);
    auto corrected = old;
    // A displaced mass at rest on a body that does not accelerate: the couple reads a design point at
    // rest and is nothing too.
    old.q = 0.2;
    corrected.q = 0.2;
    auto identical = true;
    for (auto step = 0; step < 720; step++)
    {
        const auto a = advance(body, old, Scheme::Old, tick);
        const auto b = advance(body, corrected, Scheme::Corrected, tick);
        identical = identical && old.chassis.position == corrected.chassis.position &&
                    old.chassis.linearVelocity == corrected.chassis.linearVelocity &&
                    old.chassis.angularMomentum == corrected.chassis.angularMomentum && old.q == corrected.q &&
                    old.rate == corrected.rate && b.couple == glm::dvec3(0.0) && a.reaction == b.reaction;
    }
    std::printf("\n=== zero-term control: 720 ticks bit-identical: %s ===\n", identical ? "yes" : "NO");
    REQUIRE(identical);
}

TEST_CASE("the element reaction converges to the exact system with the timestep", "[physics][suspension][kinematic-reaction]")
{
    std::printf("\n=== timestep sweep, the arc and the bead, 1 s each ===\n");
    for (const auto which : {0, 1})
    {
        const auto body = golfLike(which == 0 ? arc(0.4) : uniformLine());
        const auto start = which == 0 ? primed(body, glm::dvec3(0.0), glm::dvec3(0.0), 0.0, 3.0)
                                      : primed(body, glm::dvec3(0.0), glm::dvec3(0.0, 0.0, 1.5), 0.3, 1.0);
        const auto initial = worldOf(body, start);
        const auto momentumScale = body.unsprungMass * (which == 0 ? 1.2 : 1.0);
        const auto angularScale = which == 0 ? body.unsprungMass * 0.16 * 3.0 : glm::length(initial.angularMomentum);
        const auto seconds = which == 0 ? 1.0 : 0.5;
        auto last = RunErrors{.momentum = 1e300, .angularMomentum = 1e300, .energy = 1e300, .massPosition = 1e300};
        for (const auto hertz : {120.0, 360.0, 720.0, 2880.0})
        {
            const auto dt = 1.0 / hertz;
            const auto old = run(body, start, Scheme::Old, dt, seconds, momentumScale, angularScale);
            const auto corrected = run(body, start, Scheme::Corrected, dt, seconds, momentumScale, angularScale);
            std::printf("  %-12s %6.0f Hz: OLD |ΔP| %.2e |ΔH| %.2e |ΔE| %.2e |x_u − ref| %.2e | CORRECTED |ΔP| %.2e |ΔH| %.2e |ΔE| %.2e (end %+.2e) |x_u − ref| %.2e |ẋ_u − ref| %.2e |x_s − ref| %.2e\n",
                        body.mapping.name, hertz, old.momentum, old.angularMomentum, old.energy, old.massPosition,
                        corrected.momentum, corrected.angularMomentum, corrected.energy, corrected.energyEnd, corrected.massPosition,
                        corrected.velocity, corrected.sprungPosition);
            REQUIRE(corrected.momentum <= last.momentum);
            REQUIRE(corrected.angularMomentum <= last.angularMomentum + 1e-15);
            REQUIRE(corrected.energy <= last.energy);
            REQUIRE(corrected.massPosition <= last.massPosition);
            last = corrected;
        }
        REQUIRE(last.momentum < 5e-3);
        REQUIRE(last.massPosition < 1e-3);
    }
}

// =================================================================================================
// The whole car alone in the air
// =================================================================================================

TEST_CASE("the Golf alone in the air keeps its momentum, its angular momentum and its energy with the reaction, and Newton closes on every wheel",
          "[physics][suspension][kinematic-reaction]")
{
    // The isolated Golf — springs at zero load, no damper, no bar, no stops, no air — thrown into the
    // air with a spin and its wheels moving, 2 s. Gravity is the one external force, so the true
    // momentum changes by exactly `M·g·dt` a tick, the true angular momentum about the origin by
    // `x_com × M·g·dt` with the **true** centre of mass, and in the frame falling with the car the
    // energy is kept. For each wheel Newton is checked from the world: `m·a_u`, reconstructed from
    // the change of the wheel's whole world velocity, against `m·g` plus the force the corner handed
    // the chassis, sign reversed.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    struct Result
    {
        double momentum = 0.0;
        double angularMomentum = 0.0;
        double energy = 0.0;
        double newton = 0.0;         // worst |m·a_u − m·g + reaction| over the wheels and ticks, N
        double newtonOld = 0.0;      // the same residual with the build before's reaction alone, for the breakdown
        double kinematicMax = 0.0;   // the largest kinematic force, N
        double coupleMax = 0.0;
        double rateMax = 0.0;
        double attachment = 0.0;
    };

    std::printf("\n=== the Golf alone in the air, 2 s at 360 Hz: thrown up at 4 m/s, spun at (0.6, 0.3, 0.8) rad/s, wheels launched ===\n");
    auto results = std::array<Result, 2>{};
    for (const auto arm : {std::size_t{0}, std::size_t{1}})
    {
        const auto setup = isolatedGolf(arm == 1, true);
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
        state.chassis.linearVelocity = glm::dvec3(0.0, 4.0, 0.0);
        primeMassProperties(setup, state, world.value());
        setAngularVelocity(state.chassis, glm::dvec3(0.6, 0.3, 0.8));
        state.corners[0].wishboneRate = 2.0;
        state.corners[1].wishboneRate = -1.5;
        state.corners[2].wishboneRate = 1.0;
        state.corners[3].wishboneRate = 2.5;
        state.corners[2].wishboneAngle = 0.05;
        state.corners[3].wishboneAngle = -0.05;

        auto stepped = VehicleStep{};
        {
            const auto primed = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), 1e-12);
            REQUIRE(primed.has_value());
            stepped = primed.value();
        }
        const auto origin = state.chassis.position;
        const auto initial = carWorldOf(setup, state, stepped, origin);
        const auto g = glm::dvec3(0.0, -gravity, 0.0);
        auto momentum = initial.momentum;
        auto angular = initial.angularMomentum;
        auto& r = results[arm];
        const auto velocityAtStart = state.chassis.linearVelocity;
        const auto momentumScale = glm::length(initial.momentum);
        const auto angularScale = glm::length(initial.angularMomentum);

        // Energy in the frame falling with the car: the kinetic energy at velocities relative to free
        // fall, plus the springs'.
        const auto fallingEnergy = [&](const CarWorld& world_, const double elapsed)
        {
            const auto fall = velocityAtStart + g * elapsed;
            const auto sprung = computeMassProperties(setup.sprung).value();
            const auto omega = angularVelocity(state.chassis);
            const auto R = glm::mat3_cast(state.chassis.orientation);
            const auto sprungVelocity = state.chassis.linearVelocity +
                                        glm::cross(omega, bodyToWorld(state.chassis, sprung.centreOfMass) - state.chassis.position) - fall;
            auto kinetic = 0.5 * sprung.mass * glm::dot(sprungVelocity, sprungVelocity) +
                           0.5 * glm::dot(omega, (R * sprung.inertia * glm::transpose(R)) * omega);
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto v = world_.wheelVelocity[index] - fall;
                kinetic += 0.5 * setup.corners[index].unsprungMass * glm::dot(v, v);
            }
            return kinetic + world_.potential;
        };
        const auto initialEnergy = fallingEnergy(initial, 0.0);

        for (auto step = 0; step < 720; step++)
        {
            const auto before = carWorldOf(setup, state, stepped, origin);
            const auto beforeState = state;
            const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(result.has_value());
            stepped = result.value();
            const auto after = carWorldOf(setup, state, stepped, origin);

            // The external impulse: gravity on every mass, and its torque about the origin through the
            // true centre of mass at the start of the tick.
            momentum += state.chassis.mass * g * tick;
            angular += glm::cross(before.centreOfMass - origin, state.chassis.mass * g) * tick;
            r.momentum = std::max(r.momentum, glm::length(after.momentum - momentum) / momentumScale);
            r.angularMomentum = std::max(r.angularMomentum, glm::length(after.angularMomentum - angular) / angularScale);

            const auto energy = fallingEnergy(after, static_cast<double>(step + 1) * tick);
            const auto kinetic = energy - after.potential;
            r.energy = std::max(r.energy, std::abs(energy - initialEnergy) / initialEnergy);

            if (step < 3 || step == 10 || step == 100 || step == 359 || step == 719)
            {
                std::printf("    tick %3d: E %.3f J, q̇ (%.3f %.3f %.3f %.3f), q (%.4f %.4f %.4f %.4f), limits %d%d%d%d, |reaction| max %.1f N, |F_kin| (%.1f %.1f %.1f %.1f)\n",
                            step + 1, kinetic + after.potential, state.corners[0].wishboneRate, state.corners[1].wishboneRate,
                            state.corners[2].wishboneRate, state.corners[3].wishboneRate, state.corners[0].wishboneAngle,
                            state.corners[1].wishboneAngle, state.corners[2].wishboneAngle, state.corners[3].wishboneAngle,
                            stepped.corners[0].rangeLimit != raceengine::RangeLimit::None, stepped.corners[1].rangeLimit != raceengine::RangeLimit::None,
                            stepped.corners[2].rangeLimit != raceengine::RangeLimit::None, stepped.corners[3].rangeLimit != raceengine::RangeLimit::None,
                            std::max({glm::length(stepped.corners[0].chassisReaction), glm::length(stepped.corners[1].chassisReaction),
                                      glm::length(stepped.corners[2].chassisReaction), glm::length(stepped.corners[3].chassisReaction)}),
                            glm::length(stepped.corners[0].kinematicForce), glm::length(stepped.corners[1].kinematicForce),
                            glm::length(stepped.corners[2].kinematicForce), glm::length(stepped.corners[3].kinematicForce));
            }
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto& solution = stepped.corners[index];
                const auto m = setup.corners[index].unsprungMass;
                const auto acceleration = (after.wheelVelocity[index] - before.wheelVelocity[index]) / tick;

                // Newton on the wheel from the world. The chassis body carries the wheel's weight and
                // the inertial force of its design point itself, so what the corner hands the chassis
                // is `−m·(a_u − a_design)`, and Newton for the wheel reads `m·a_u = m·a_design − reaction`
                // with gravity on both sides. `a_design` from what the body actually did this tick.
                const auto omega0 = angularVelocity(beforeState.chassis);
                const auto alpha0 = (angularVelocity(state.chassis) - omega0) / tick;
                const auto designArm = bodyToWorld(beforeState.chassis, setup.corners[index].hardpoints.wheelCentre) - beforeState.chassis.position;
                const auto designAcceleration = (state.chassis.linearVelocity - beforeState.chassis.linearVelocity) / tick +
                                                glm::cross(alpha0, designArm) + glm::cross(omega0, glm::cross(omega0, designArm));
                const auto residual = m * acceleration - m * designAcceleration + solution.chassisReaction;
                if (index == 0 && (step == 1 || step == 100))
                {
                    std::printf("      FL tick %d: m·a_u (%.1f %.1f %.1f) N, m·a_design (%.1f %.1f %.1f), reaction handed (%.1f %.1f %.1f), of which kinematic (%.1f %.1f %.1f): residual %.2f N\n",
                                step + 1, m * acceleration.x, m * acceleration.y, m * acceleration.z, m * designAcceleration.x,
                                m * designAcceleration.y, m * designAcceleration.z, solution.chassisReaction.x, solution.chassisReaction.y,
                                solution.chassisReaction.z, solution.kinematicForce.x, solution.kinematicForce.y, solution.kinematicForce.z,
                                glm::length(residual));
                }
                r.newton = std::max(r.newton, glm::length(residual));
                r.newtonOld = std::max(r.newtonOld, glm::length(residual - solution.kinematicForce));
                r.kinematicMax = std::max(r.kinematicMax, glm::length(solution.kinematicForce));
                r.coupleMax = std::max(r.coupleMax, glm::length(solution.kinematicCouple));
                r.rateMax = std::max(r.rateMax, std::abs(beforeState.corners[index].wishboneRate));
                const auto armWorld = bodyToWorld(beforeState.chassis, solution.suspension.wheelCentre) - beforeState.chassis.position;
                const auto actual = attachmentAcceleration(
                    ChassisMotion{.acceleration = (state.chassis.linearVelocity - beforeState.chassis.linearVelocity) / tick,
                                  .angularVelocity = omega0,
                                  .angularAcceleration = alpha0},
                    armWorld);
                r.attachment = std::max(r.attachment, glm::length(solution.attachmentAcceleration - actual));
            }
        }

        std::printf("  %s: |ΔP|/P₀ %.2e, |ΔH|/H₀ %.2e, |ΔE|/E₀ %.2e; Newton on the wheels: with the reaction as handed %.2e N, with −m·R·C'·q̈ alone %.1f N; "
                    "|F_kin| max %.1f N, couple max %.2f N·m, |q̇| max %.2f rad/s, |a assumed − actual| %.1e m/s²\n",
                    arm == 1 ? "CORRECTED" : "OLD      ", r.momentum, r.angularMomentum, r.energy, r.newton, r.newtonOld, r.kinematicMax,
                    r.coupleMax, r.rateMax, r.attachment);
    }

    // The build before: momentum and angular momentum drift by the omitted terms; the corrected car
    // keeps them to the explicit step's first order and Newton closes on every wheel to what a one-tick
    // difference of velocities can say about an acceleration.
    REQUIRE(results[0].momentum > 5.0 * results[1].momentum);
    REQUIRE(results[0].angularMomentum > 5.0 * results[1].angularMomentum);
    REQUIRE(results[1].energy < 1e-2);
    REQUIRE(results[1].newton < 25.0);
    REQUIRE(results[1].newtonOld > 50.0);
    REQUIRE(results[1].newton < 0.1 * results[1].newtonOld);
    REQUIRE(results[1].attachment < 2e-2);
}

TEST_CASE("the Golf's momentum closes with the timestep in the air", "[physics][suspension][kinematic-reaction]")
{
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());
    const auto setup = isolatedGolf(true, true);
    const auto g = glm::dvec3(0.0, -gravity, 0.0);

    std::printf("\n=== the Golf alone in the air by timestep, 1 s ===\n");
    auto lastMomentum = 1e300;
    auto lastAngular = 1e300;
    for (const auto hertz : {120.0, 360.0, 720.0, 2880.0})
    {
        const auto dt = 1.0 / hertz;
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
        state.chassis.linearVelocity = glm::dvec3(0.0, 4.0, 0.0);
        primeMassProperties(setup, state, world.value());
        setAngularVelocity(state.chassis, glm::dvec3(0.6, 0.3, 0.8));
        state.corners[0].wishboneRate = 2.0;
        state.corners[1].wishboneRate = -1.5;
        state.corners[2].wishboneRate = 1.0;
        state.corners[3].wishboneRate = 2.5;
        auto stepped = VehicleStep{};
        {
            const auto primed = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), 1e-12);
            REQUIRE(primed.has_value());
            stepped = primed.value();
        }
        const auto origin = state.chassis.position;
        const auto initial = carWorldOf(setup, state, stepped, origin);
        auto momentum = initial.momentum;
        auto angular = initial.angularMomentum;
        auto worstMomentum = 0.0;
        auto worstAngular = 0.0;
        const auto ticks = static_cast<int>(std::lround(hertz));
        for (auto step = 0; step < ticks; step++)
        {
            const auto before = carWorldOf(setup, state, stepped, origin);
            const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), dt);
            REQUIRE(result.has_value());
            stepped = result.value();
            const auto after = carWorldOf(setup, state, stepped, origin);
            momentum += state.chassis.mass * g * dt;
            angular += glm::cross(before.centreOfMass - origin, state.chassis.mass * g) * dt;
            worstMomentum = std::max(worstMomentum, glm::length(after.momentum - momentum) / glm::length(initial.momentum));
            worstAngular = std::max(worstAngular, glm::length(after.angularMomentum - angular) / glm::length(initial.angularMomentum));
        }
        std::printf("  %6.0f Hz: |ΔP|/P₀ %.2e, |ΔH|/H₀ %.2e\n", hertz, worstMomentum, worstAngular);
        REQUIRE(worstMomentum <= lastMomentum * 1.001);
        REQUIRE(worstAngular <= lastAngular * 1.001);
        lastMomentum = worstMomentum;
        lastAngular = worstAngular;
    }
    REQUIRE(lastMomentum < 1e-4);
}

TEST_CASE("both rear wheels launched alike: the kinematic reactions' lateral parts and roll couple cancel, the vertical parts and pitch reinforce",
          "[physics][suspension][kinematic-reaction]")
{
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());
    const auto setup = isolatedGolf(true, true);
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
    primeMassProperties(setup, state, world.value());
    state.corners[2].wishboneRate = 3.0;
    state.corners[3].wishboneRate = 3.0;

    const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
    REQUIRE(result.has_value());
    const auto& left = result->corners[2];
    const auto& right = result->corners[3];
    const auto leftArm = bodyToWorld(state.chassis, left.suspension.wheelCentre) - state.chassis.position;
    const auto rightArm = bodyToWorld(state.chassis, right.suspension.wheelCentre) - state.chassis.position;
    const auto leftTorque = glm::cross(leftArm, left.kinematicForce) + left.kinematicCouple;
    const auto rightTorque = glm::cross(rightArm, right.kinematicForce) + right.kinematicCouple;

    std::printf("\n=== both rears at 3 rad/s, body still: kinematic force RL (%.2f, %.2f, %.2f) RR (%.2f, %.2f, %.2f) N; torque about the centre RL (%.3f, %.3f, %.3f) RR (%.3f, %.3f, %.3f) N·m ===\n",
                left.kinematicForce.x, left.kinematicForce.y, left.kinematicForce.z, right.kinematicForce.x, right.kinematicForce.y,
                right.kinematicForce.z, leftTorque.x, leftTorque.y, leftTorque.z, rightTorque.x, rightTorque.y, rightTorque.z);

    // Mirror symmetry across the car's centre plane (x): lateral force and roll (z) and yaw (y) torque cancel; vertical
    // and longitudinal force and pitch (x) torque reinforce.
    REQUIRE(left.kinematicForce.x == Catch::Approx(-right.kinematicForce.x).margin(1e-9));
    REQUIRE(left.kinematicForce.y == Catch::Approx(right.kinematicForce.y).margin(1e-9));
    REQUIRE(left.kinematicForce.z == Catch::Approx(right.kinematicForce.z).margin(1e-9));
    REQUIRE(leftTorque.x == Catch::Approx(rightTorque.x).margin(1e-9));
    REQUIRE(leftTorque.y == Catch::Approx(-rightTorque.y).margin(1e-9));
    REQUIRE(leftTorque.z == Catch::Approx(-rightTorque.z).margin(1e-9));
    // And the centripetal pull is upward-ish and of the size m·|C''|·q̇² at 3 rad/s.
    REQUIRE(glm::length(left.kinematicForce) == Catch::Approx(setup.corners[2].unsprungMass * glm::length(cornerJacobianCurvature(setup.corners[2], 0.0, 0.0, true)) * 9.0).epsilon(0.05));
}

TEST_CASE("bead on a rod, the body a million tonnes: the corner alone against the closed form", "[.kinematic-reaction-probe]")
{
    // With the body immovable the bead's motion is `q = q₀·cosh(ωt) + (q̇₀/ω)·sinh(ωt)` exactly and the
    // body's response is nothing: whatever error is left is the corner side's.
    for (const auto sprung : {1200.0, 1.0e7})
    {
        auto body = golfLike(uniformLine());
        body.sprungMass = sprung;
        if (sprung > 1e6)
        {
            body.sprungInertia = glm::dmat3(1.0) * 1.0e7;
        }
        const auto omega = 1.5;
        for (const auto hertz : {360.0, 2880.0, 11520.0})
        {
            const auto dt = 1.0 / hertz;
            auto state = primed(body, glm::dvec3(0.0), glm::dvec3(0.0, 0.0, omega), 0.3, 1.0);
            auto exact = exactOf(body, state);
            const auto ticks = static_cast<int>(std::lround(0.5 * hertz));
            for (auto step = 0; step < ticks; step++)
            {
                (void)advance(body, state, Scheme::Corrected, dt);
                for (auto sub = 0; sub < 100; sub++)
                {
                    rungeKutta(body, exact, dt / 100.0);
                }
            }
            const auto t = 0.5;
            const auto closed = 0.3 * std::cosh(omega * t) + (1.0 / omega) * std::sinh(omega * t);
            const auto closedRate = 0.3 * omega * std::sinh(omega * t) + std::cosh(omega * t);
            std::printf("  M_s %.0e, %6.0f Hz: engine q %.6f q̇ %.6f | reference q %.6f q̇ %.6f | closed form (fixed body) q %.6f q̇ %.6f | ω engine %.6f ref %.6f\n",
                        sprung, hertz, state.q, state.rate, exact.q, exact.rate, closed, closedRate, angularVelocity(state.chassis).z, exact.omega.z);
        }
    }
}

TEST_CASE("the Golf falling with the frame term off: each wheel's momentum change against what the chassis was handed", "[.kinematic-reaction-probe]")
{
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());
    auto setup = golfGtiMk7().value();
    setup.aero.clear();
    setup.frameAcceleration = false;
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 30.0, 200.0);
    primeMassProperties(setup, state, world.value());
    const auto g = glm::dvec3(0.0, -gravity, 0.0);
    for (auto step = 0; step < 8; step++)
    {
        const auto before = state;
        const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(result.has_value());
        const auto chassisImpulse = state.chassis.mass * (state.chassis.linearVelocity - before.chassis.linearVelocity) - state.chassis.mass * g * tick;
        auto wheels = glm::dvec3(0.0);
        auto handed = glm::dvec3(0.0);
        std::printf("  tick %d: M·Δv − M·g·dt (%.3e %.3e %.3e)\n", step + 1, chassisImpulse.x, chassisImpulse.y, chassisImpulse.z);
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            const auto& solution = result->corners[index];
            const auto s0 = solveCornerWithJacobian(corner.hardpoints, before.corners[index].wishboneAngle, 0.0).value();
            const auto s1 = solveCornerWithJacobian(corner.hardpoints, state.corners[index].wishboneAngle, 0.0).value();
            const auto w0 = angularVelocity(before.chassis);
            const auto w1 = angularVelocity(state.chassis);
            const auto p0 = corner.unsprungMass * (glm::cross(w0, before.chassis.orientation * (s0.wheelCentre - corner.hardpoints.wheelCentre)) +
                                                   before.chassis.orientation * s0.wheelCentrePerAngle * before.corners[index].wishboneRate);
            const auto p1 = corner.unsprungMass * (glm::cross(w1, state.chassis.orientation * (s1.wheelCentre - corner.hardpoints.wheelCentre)) +
                                                   state.chassis.orientation * s1.wheelCentrePerAngle * state.corners[index].wishboneRate);
            const auto change = p1 - p0;
            const auto impulse = -solution.chassisReaction * tick;
            wheels += change;
            handed += impulse;
            const auto b0 = before.chassis.orientation * s0.wheelCentrePerAngle;
            const auto b1 = state.chassis.orientation * s1.wheelCentrePerAngle;
            const auto k = state.chassis.orientation * raceengine::cornerJacobianCurvature(corner, before.corners[index].wishboneAngle, 0.0, true);
            const auto dq = state.corners[index].wishboneAngle - before.corners[index].wishboneAngle;
            std::printf("    %s q̇ %.4f -> %.4f: Δp (%.3e %.3e %.3e), −F·dt (%.3e %.3e %.3e), diff (%.2e %.2e %.2e); (b1−b0) (%.2e %.2e %.2e) vs k·Δq (%.2e %.2e %.2e), |ω| %.1e\n",
                        cornerAbbreviation(static_cast<Corner>(index)), before.corners[index].wishboneRate, state.corners[index].wishboneRate,
                        change.x, change.y, change.z, impulse.x, impulse.y, impulse.z, (change - impulse).x, (change - impulse).y, (change - impulse).z,
                        (b1 - b0).x, (b1 - b0).y, (b1 - b0).z, (k * dq).x, (k * dq).y, (k * dq).z, glm::length(w0));
        }
        std::printf("    Σ wheels Δp (%.3e %.3e %.3e), Σ handed (%.3e %.3e %.3e), total residual (%.2e %.2e %.2e)\n", wheels.x, wheels.y, wheels.z,
                    handed.x, handed.y, handed.z, (chassisImpulse + wheels).x, (chassisImpulse + wheels).y, (chassisImpulse + wheels).z);
    }
}
