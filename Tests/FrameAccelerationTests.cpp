// The chassis's own acceleration in the corner's equation (docs/frame-acceleration-brief.md,
// 2026-09-08 latest of all): the proofs.
//
// The corner's coordinate `q` is relative to the chassis: the wheel centre is `x_B + R·(C(q) − c)`,
// so the unsprung mass's absolute acceleration is the acceleration of the body-fixed point under it —
// `a_B + α × r + ω × (ω × r)` — plus `R·C'·q̈`, plus `R·C''·q̇²`, plus a Coriolis term `2ω × R·C'·q̇`.
// Newton's law for the mass, projected onto the one direction it can move in relative to the body,
// `R·C'`, is
//
//     m_u·|C'|²·q̈ = Q_elements + Q_tyre − m_u·(R·C')·(a_attach − g) − m_u·(C'·C'')·q̇²
//
// with the Coriolis term gone (it is perpendicular to `R·C'`). Until this change the corner carried
// the weight, `−m_u·g·C'_y`, and nothing for `a_attach`: the chassis was, to every corner, infinitely
// massive. That is why a wheel hanging in free fall pulled the body down with its whole weight (the
// range-constraint brief's 629 N at the Golf's rear droop limit, where the spring less the stop is
// 207) and why a wheel under a body accelerating at 1 g was weighed wrong by exactly its weight.
//
// The chassis's own equation — the whole car's mass on the body, the reaction `−m_u·q̈·R·C'` at the
// wheel centre — was already the exact one, which is why the momentum ledger closed before this and
// still does. What the fix adds is the other half of a symmetric mass matrix: the corner's equation
// carries `−m_u·(R·C')·a_attach`, and because `a_attach` itself carries the four reactions, the four
// velocity steps are solved together, coupled through the rigid body's mobility at the wheel centres
// (`attachmentMobility`). A corner's inertia against a body that gives way is then the reduced one.
//
// Every table prints; `./EngineTests "[frame-acceleration]" -s` shows them.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <expected>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::AeroSurface;
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
using raceengine::damperElementOf;
using raceengine::ForceAccumulator;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::integrate;
using raceengine::linearDamper;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::RangeLimit;
using raceengine::RigidBodyState;
using raceengine::setAngularVelocity;
using raceengine::solveCornerWithJacobian;
using raceengine::solveElement;
using raceengine::springElementOf;
using raceengine::springFreeLengthForLoad;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::worldInverseInertia;

namespace
{

constexpr auto gravity = 9.80665;
constexpr auto tick = 1.0 / 360.0;
constexpr auto degrees = 180.0 / 3.14159265358979323846;

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

// The Golf with no air on it, so that gravity is the one external force in the air.
[[nodiscard]] VehicleSetup golfInVacuum(const bool frameAcceleration)
{
    auto setup = golfGtiMk7().value();
    setup.aero.clear();
    setup.frameAcceleration = frameAcceleration;
    return setup;
}

// The Golf with every corner at a relative equilibrium at its design position and nothing else on
// the corner: the spring's free length re-solved for zero load, no damper, no seal friction, no bar,
// and — unless asked to keep them — both stops out of reach. In the air with `q = 0` and `q̇ = 0`
// this car has no reason to move its wheels relative to its body, and whether it does is the whole
// question.
[[nodiscard]] VehicleSetup unloadedGolf(const bool frameAcceleration, const bool damped = false,
                                        const bool authoredStops = false)
{
    auto setup = golfInVacuum(frameAcceleration);
    for (auto& corner : setup.corners)
    {
        corner.springFreeLength = springFreeLengthForLoad(corner, 0.0).value();
        if (!damped)
        {
            corner.damper = linearDamper(0.0, 0.0);
        }
        corner.damperFriction = 0.0;
        corner.damperFrictionShape = raceengine::Curve{};
        corner.antiRollRate = 0.0;
        if (!authoredStops)
        {
            corner.bumpStop.gap = 1.0;
            corner.droopStop.gap = 1.0;
        }
    }
    return setup;
}

// One tick with a dt too small to move anything, so the chassis carries its mass properties and
// the tests can read them — the way the sandbox reads the centre of mass.
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

// The momentum ledger of one tick, as the range-constraint proofs keep it, plus the acceleration
// each corner assumed of its attachment against the one the chassis then had there.
struct Ledger
{
    glm::dvec3 residual{0.0};
    std::array<double, cornerCount> accelerationError{};
    std::array<double, cornerCount> attachmentError{}; // |a assumed − a actual| at the wheel centre, m/s²
};

// The car's true momentum from a state: the chassis's, plus every unsprung mass's beyond its design
// point — `m·[ω × (r − r_design) + R·C'·q̇]`, the wheel centre and Jacobian from a fresh solve at the
// corner's angle (2026-09-08 latest of all (d), docs/chassis-kinematic-reaction-brief.md). Before (d)
// the ledger carried the `R·C'·q̇` part alone, which was the whole of what the chassis was handed.
[[nodiscard]] glm::dvec3 trueMomentum(const VehicleSetup& setup, const VehicleState& state)
{
    const auto omega = angularVelocity(state.chassis);
    auto momentum = state.chassis.mass * state.chassis.linearVelocity;
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        const auto solved = solveCornerWithJacobian(corner.hardpoints, state.corners[index].wishboneAngle, 0.0).value();
        const auto offset = state.chassis.orientation * (solved.wheelCentre - corner.hardpoints.wheelCentre);
        momentum += corner.unsprungMass * (glm::cross(omega, offset) +
                                           state.chassis.orientation * solved.wheelCentrePerAngle * state.corners[index].wishboneRate);
    }
    return momentum;
}

[[nodiscard]] Ledger stepLedger(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world,
                                const double dt, VehicleStep& stepped)
{
    const auto before = state;
    const auto momentumBefore = trueMomentum(setup, state);
    const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, dt);
    REQUIRE(result.has_value());
    stepped = result.value();

    auto ledger = Ledger{};
    ledger.residual = trueMomentum(setup, state) - momentumBefore - glm::dvec3(0.0, -gravity * state.chassis.mass * dt, 0.0);

    const auto omegaBefore = angularVelocity(before.chassis);
    const auto omegaAfter = angularVelocity(state.chassis);
    const auto alpha = (omegaAfter - omegaBefore) / dt;
    const auto linear = (state.chassis.linearVelocity - before.chassis.linearVelocity) / dt;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& solution = stepped.corners[index];
        const auto deltaRate = state.corners[index].wishboneRate - before.corners[index].wishboneRate;

        if (solution.patch.inContact)
        {
            ledger.residual -= (solution.forces.tireVertical * solution.patch.normal +
                                solution.contact.tyre.longitudinal * solution.contact.forward +
                                solution.contact.tyre.lateral * solution.contact.lateral) *
                               dt;
        }
        ledger.accelerationError[index] =
            std::abs(solution.generalisedForce / solution.generalisedInertia * dt - deltaRate);

        const auto arm = bodyToWorld(before.chassis, solution.suspension.wheelCentre) - before.chassis.position;
        const auto actual = attachmentAcceleration(
            ChassisMotion{.acceleration = linear, .angularVelocity = omegaBefore, .angularAcceleration = alpha}, arm);
        ledger.attachmentError[index] = glm::length(solution.attachmentAcceleration - actual);
    }

    return ledger;
}

// --- a 4 × 4 symmetric generalised eigenproblem, for the modal references ------------------------

using Matrix = std::array<std::array<double, cornerCount>, cornerCount>;
using Vector = std::array<double, cornerCount>;

struct Modes
{
    Vector omega{};                    // rad/s, ascending
    std::array<Vector, cornerCount> x; // the mode shapes, x[k][i] is corner i's share of mode k
};

// K·x = ω²·A·x with A symmetric positive definite: Cholesky A = L·Lᵀ, C = L⁻¹·K·L⁻ᵀ, Jacobi on C.
[[nodiscard]] Modes solveModes(const Matrix& a, const Matrix& k)
{
    auto l = Matrix{};
    for (auto i = std::size_t{0}; i < cornerCount; i++)
    {
        for (auto j = std::size_t{0}; j <= i; j++)
        {
            auto sum = a[i][j];
            for (auto p = std::size_t{0}; p < j; p++)
            {
                sum -= l[i][p] * l[j][p];
            }
            l[i][j] = i == j ? std::sqrt(sum) : sum / l[j][j];
        }
    }

    // Y = L⁻¹·K, then C = Y·L⁻ᵀ, both by forward substitution.
    auto y = Matrix{};
    for (auto col = std::size_t{0}; col < cornerCount; col++)
    {
        for (auto i = std::size_t{0}; i < cornerCount; i++)
        {
            auto sum = k[i][col];
            for (auto p = std::size_t{0}; p < i; p++)
            {
                sum -= l[i][p] * y[p][col];
            }
            y[i][col] = sum / l[i][i];
        }
    }
    auto c = Matrix{};
    for (auto row = std::size_t{0}; row < cornerCount; row++)
    {
        for (auto j = std::size_t{0}; j < cornerCount; j++)
        {
            auto sum = y[row][j];
            for (auto p = std::size_t{0}; p < j; p++)
            {
                sum -= l[j][p] * c[row][p];
            }
            c[row][j] = sum / l[j][j];
        }
    }

    auto v = Matrix{};
    for (auto i = std::size_t{0}; i < cornerCount; i++)
    {
        v[i][i] = 1.0;
    }
    for (auto sweep = 0; sweep < 100; sweep++)
    {
        auto off = 0.0;
        for (auto p = std::size_t{0}; p < cornerCount; p++)
        {
            for (auto q = p + 1; q < cornerCount; q++)
            {
                off += c[p][q] * c[p][q];
            }
        }
        if (off < 1e-30)
        {
            break;
        }
        for (auto p = std::size_t{0}; p < cornerCount; p++)
        {
            for (auto q = p + 1; q < cornerCount; q++)
            {
                if (std::abs(c[p][q]) < 1e-300)
                {
                    continue;
                }
                const auto theta = 0.5 * std::atan2(2.0 * c[p][q], c[q][q] - c[p][p]);
                const auto cs = std::cos(theta);
                const auto sn = std::sin(theta);
                for (auto r = std::size_t{0}; r < cornerCount; r++)
                {
                    const auto crp = c[r][p];
                    const auto crq = c[r][q];
                    c[r][p] = cs * crp - sn * crq;
                    c[r][q] = sn * crp + cs * crq;
                }
                for (auto r = std::size_t{0}; r < cornerCount; r++)
                {
                    const auto cpr = c[p][r];
                    const auto cqr = c[q][r];
                    c[p][r] = cs * cpr - sn * cqr;
                    c[q][r] = sn * cpr + cs * cqr;
                }
                for (auto r = std::size_t{0}; r < cornerCount; r++)
                {
                    const auto vrp = v[r][p];
                    const auto vrq = v[r][q];
                    v[r][p] = cs * vrp - sn * vrq;
                    v[r][q] = sn * vrp + cs * vrq;
                }
            }
        }
    }

    // x = L⁻ᵀ·v, by back substitution, one eigenvector per column.
    auto modes = Modes{};
    auto order = std::array<std::size_t, cornerCount>{0, 1, 2, 3};
    std::sort(order.begin(), order.end(), [&](const auto lhs, const auto rhs) { return c[lhs][lhs] < c[rhs][rhs]; });
    for (auto slot = std::size_t{0}; slot < cornerCount; slot++)
    {
        const auto col = order[slot];
        modes.omega[slot] = std::sqrt(std::max(c[col][col], 0.0));
        for (auto i = cornerCount; i-- > 0;)
        {
            auto sum = v[i][col];
            for (auto p = i + 1; p < cornerCount; p++)
            {
                sum -= l[p][i] * modes.x[slot][p];
            }
            modes.x[slot][i] = sum / l[i][i];
        }
    }
    return modes;
}

// The linearised airborne car at its design position: the corners' spring stiffness in their own
// coordinates, their inertias, and the chassis coupling between them at the current attitude.
struct Linearised
{
    Matrix inertia{};    // the corners' own inertias on the diagonal, less the coupling when asked
    Matrix stiffness{};  // the springs', diagonal
    Vector jacobianY{};  // travelPerAngle per corner
};

[[nodiscard]] Linearised linearise(const VehicleSetup& setup, const VehicleState& primed, const bool coupled)
{
    auto out = Linearised{};
    auto jacobian = std::array<glm::dvec3, cornerCount>{};
    auto arm = std::array<glm::dvec3, cornerCount>{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0).value();
        const auto spring = solveElement(corner.hardpoints, springElementOf(corner.hardpoints), 0.0);
        out.stiffness[index][index] = corner.springRate * spring.lengthPerAngle * spring.lengthPerAngle;
        out.inertia[index][index] = corner.unsprungMass * glm::dot(design.wheelCentrePerAngle, design.wheelCentrePerAngle);
        out.jacobianY[index] = design.travelPerAngle;
        jacobian[index] = primed.chassis.orientation * design.wheelCentrePerAngle;
        arm[index] = bodyToWorld(primed.chassis, design.wheelCentre) - primed.chassis.position;
    }

    if (coupled)
    {
        const auto inverseInertia = worldInverseInertia(primed.chassis);
        for (auto i = std::size_t{0}; i < cornerCount; i++)
        {
            for (auto j = std::size_t{0}; j < cornerCount; j++)
            {
                out.inertia[i][j] -= setup.corners[i].unsprungMass * setup.corners[j].unsprungMass *
                                     attachmentMobility(primed.chassis.mass, inverseInertia, arm[j], jacobian[j],
                                                        arm[i], jacobian[i]);
            }
        }
    }
    return out;
}

// The period of a record from its zero crossings, linearly interpolated, over every full cycle it
// holds; NaN with fewer than three crossings.
[[nodiscard]] double periodOf(const std::vector<double>& record, const double dt)
{
    auto crossings = std::vector<double>{};
    for (auto index = std::size_t{1}; index < record.size(); index++)
    {
        if ((record[index - 1] < 0.0) != (record[index] < 0.0))
        {
            const auto fraction = record[index - 1] / (record[index - 1] - record[index]);
            crossings.push_back((static_cast<double>(index - 1) + fraction) * dt);
        }
    }
    if (crossings.size() < 3)
    {
        return std::nan("");
    }
    const auto cycles = (crossings.size() - 1) / 2;
    return (crossings[2 * cycles] - crossings[0]) / static_cast<double>(cycles);
}

// The energy of the airborne car in the frame falling with it — where gravity is gone and, with the
// corner equation right, so is the wheels' weight: the chassis's kinetic energy at its velocity
// relative to free fall, the cross term and the corners' own kinetic energy from the same kinetic
// energy the coupled equations derive from, and the springs' potential.
[[nodiscard]] double fallingFrameEnergy(const VehicleSetup& setup, const VehicleState& state, const VehicleStep& step,
                                        const glm::dvec3& velocityAtStart, const double elapsed)
{
    const auto relative = state.chassis.linearVelocity - (velocityAtStart + glm::dvec3(0.0, -gravity * elapsed, 0.0));
    const auto omega = angularVelocity(state.chassis);
    auto energy = 0.5 * state.chassis.mass * glm::dot(relative, relative) + 0.5 * glm::dot(omega, state.chassis.angularMomentum);

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        const auto& suspension = step.corners[index].suspension;
        const auto jacobian = state.chassis.orientation * suspension.wheelCentrePerAngle;
        const auto arm = bodyToWorld(state.chassis, suspension.wheelCentre) - state.chassis.position;
        const auto rate = state.corners[index].wishboneRate;
        energy += corner.unsprungMass * glm::dot(relative + glm::cross(omega, arm), jacobian) * rate +
                  0.5 * corner.unsprungMass * glm::dot(jacobian, jacobian) * rate * rate;

        const auto spring = solveElement(corner.hardpoints, springElementOf(corner.hardpoints), state.corners[index].wishboneAngle);
        const auto stretch = corner.springFreeLength - spring.length;
        energy += 0.5 * corner.springRate * stretch * stretch;
    }
    return energy;
}

[[nodiscard]] glm::dquat attitude(const double pitchDegrees, const double rollDegrees)
{
    const auto pitch = glm::angleAxis(pitchDegrees / degrees, glm::dvec3(1.0, 0.0, 0.0));
    const auto roll = glm::angleAxis(rollDegrees / degrees, glm::dvec3(0.0, 0.0, 1.0));
    return glm::normalize(pitch * roll);
}

} // namespace

// =================================================================================================
// Element proofs
// =================================================================================================

TEST_CASE("the attachment's mobility is the rigid body's own response, symmetric, and nothing on an infinitely heavy body",
          "[physics][suspension][frame-acceleration]")
{
    // A body with three different principal inertias, turned to a generic attitude.
    auto body = RigidBodyState{};
    body.mass = 1350.0;
    body.inertia = glm::dmat3(1.0);
    body.inertia[0][0] = 2100.0;
    body.inertia[1][1] = 2300.0;
    body.inertia[2][2] = 450.0;
    body.inverseInertia = glm::inverse(body.inertia);
    body.orientation = attitude(7.0, -11.0);

    const auto inverse = worldInverseInertia(body);
    const auto armA = glm::dvec3(-0.76, -0.31, 1.32);
    const auto armB = glm::dvec3(0.75, -0.29, -1.28);
    const auto jacobianA = glm::normalize(glm::dvec3(0.05, 0.31, -0.02));
    const auto jacobianB = glm::normalize(glm::dvec3(-0.04, 0.30, 0.03));

    // A unit generalised force at B along its Jacobian is the force `jacobianB` at B: the body's
    // linear acceleration is that over the mass and its angular one the inverse tensor times the
    // moment; the point at A then accelerates by `a + α × r_A`, and the mobility is its share along A.
    const auto linear = jacobianB / body.mass;
    const auto angular = inverse * glm::cross(armB, jacobianB);
    const auto atA = linear + glm::cross(angular, armA);
    const auto direct = glm::dot(jacobianA, atA);

    const auto ab = attachmentMobility(body.mass, inverse, armB, jacobianB, armA, jacobianA);
    const auto ba = attachmentMobility(body.mass, inverse, armA, jacobianA, armB, jacobianB);

    std::printf("\n=== the attachment mobility ===\n  direct %.12e  helper A<-B %.12e  helper B<-A %.12e  (1/M is %.6e)\n",
                direct, ab, ba, 1.0 / body.mass);

    REQUIRE(ab == Catch::Approx(direct).epsilon(1e-12));
    REQUIRE(ba == Catch::Approx(ab).epsilon(1e-12));

    // An infinitely heavy body does not move: the mobility is exactly zero, which is the corner
    // equation this replaces.
    REQUIRE(attachmentMobility(1.0 / 0.0, glm::dmat3(0.0), armB, jacobianB, armA, jacobianA) == 0.0);

    // And the attachment's acceleration is what the integrator does to a body-fixed point, to
    // first order in the tick, converging as the tick shrinks: a force and a torque on a body
    // already turning, the point's velocity differenced across one step.
    auto forces = ForceAccumulator{};
    forces.force = glm::dvec3(900.0, -4000.0, 1200.0);
    forces.torque = glm::dvec3(600.0, -250.0, 900.0);
    setAngularVelocity(body, glm::dvec3(0.4, -0.9, 0.7));
    body.linearVelocity = glm::dvec3(2.0, -1.0, 25.0);

    const auto omega = angularVelocity(body);
    const auto motion = ChassisMotion{.acceleration = forces.force / body.mass,
                                      .angularVelocity = omega,
                                      .angularAcceleration = inverse * (forces.torque - glm::cross(omega, body.angularMomentum))};
    const auto predicted = attachmentAcceleration(motion, armA);

    auto lastError = 1e300;
    std::printf("  attachment acceleration against the integrator: predicted (%.6f, %.6f, %.6f)\n", predicted.x,
                predicted.y, predicted.z);
    for (const auto dt : {1e-2, 1e-3, 1e-4, 1e-5})
    {
        auto after = body;
        integrate(after, forces, dt);
        const auto pointBefore = body.linearVelocity + glm::cross(omega, armA);
        const auto armAfter = after.orientation * (glm::conjugate(body.orientation) * armA);
        const auto pointAfter = after.linearVelocity + glm::cross(angularVelocity(after), armAfter);
        const auto measured = (pointAfter - pointBefore) / dt;
        const auto error = glm::length(measured - predicted);
        std::printf("    dt %.0e: measured (%.6f, %.6f, %.6f), error %.3e m/s²\n", dt, measured.x, measured.y,
                    measured.z, error);
        REQUIRE(error < lastError);
        lastError = error;
    }
    REQUIRE(lastError < 1e-3);
}

// =================================================================================================
// Free fall
// =================================================================================================

TEST_CASE("in free fall a hanging wheel pulls on its limit with the spring less the stop, and its weight is gone",
          "[physics][suspension][frame-acceleration]")
{
    // The range-constraint brief's whole-car case, in both arms: the Golf dropped from 30 m with no
    // air on it, every wheel hanging from its droop limit within a fifth of a second, the car falling
    // for two seconds. The chassis is in free fall at g; so is every wheel, so a wheel's weight
    // cannot be part of what its limit carries. The build before said it was.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    struct Arm
    {
        const char* name = "";
        bool frameAcceleration = false;
        std::array<double, cornerCount> reaction{};
        std::array<double, cornerCount> expected{}; // the spring less the stop, at the wheel, from the solution's own forces
        std::array<double, cornerCount> frame{};
        double worstResidual = 0.0;
        double worstAcceleration = 0.0;
        double worstAttachment = 0.0;
        int worstAttachmentTick = -1;
        std::uint32_t worstAttachmentSweeps = 0;
        std::string worstAttachmentState;
        double fallRateError = 0.0;
        int heldTicks = 0;
        int arrivals = 0;
        std::array<int, cornerCount> arrivalTick{-1, -1, -1, -1};
        std::uint32_t sweeps = 0;
    };

    auto arms = std::array<Arm, 2>{};
    arms[0].name = "ON";
    arms[0].frameAcceleration = true;
    arms[1].name = "OFF, the frame term before";
    arms[1].frameAcceleration = false;

    std::printf("\n=== the Golf in the air, 2 s, both arms ===\n");
    for (auto& arm : arms)
    {
        const auto setup = golfInVacuum(arm.frameAcceleration);
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 30.0, 200.0);
        primeMassProperties(setup, state, world.value());
        auto held = std::array<bool, cornerCount>{};

        for (auto step = 0; step < 720; step++)
        {
            const auto before = state.chassis.linearVelocity.y;
            auto stepped = VehicleStep{};
            const auto ledger = stepLedger(setup, state, world.value(), tick, stepped);
            if (glm::length(ledger.residual) > 1e-4)
            {
                std::printf("    tick %d: ledger residual %.3e N·s (%.2e %.2e %.2e), sweeps %u, limits %d%d%d%d, ω after (%.4f %.4f %.4f), |a assumed − actual| max %.2e, q̇ FL %.3f -> %.3f\n", step + 1,
                            glm::length(ledger.residual), ledger.residual.x, ledger.residual.y, ledger.residual.z, stepped.corners[0].frameSweeps,
                            stepped.corners[0].rangeLimit != RangeLimit::None, stepped.corners[1].rangeLimit != RangeLimit::None,
                            stepped.corners[2].rangeLimit != RangeLimit::None, stepped.corners[3].rangeLimit != RangeLimit::None,
                            angularVelocity(state.chassis).x, angularVelocity(state.chassis).y, angularVelocity(state.chassis).z,
                            std::max({ledger.attachmentError[0], ledger.attachmentError[1], ledger.attachmentError[2], ledger.attachmentError[3]}),
                            0.0, state.corners[0].wishboneRate);
            }
            arm.worstResidual = std::max(arm.worstResidual, glm::length(ledger.residual));

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                arm.worstAcceleration = std::max(arm.worstAcceleration, ledger.accelerationError[index]);
                if (arm.frameAcceleration)
                {
                    if (ledger.attachmentError[index] > arm.worstAttachment)
                    {
                        arm.worstAttachment = ledger.attachmentError[index];
                        arm.worstAttachmentTick = step + 1;
                        arm.worstAttachmentSweeps = stepped.corners[index].frameSweeps;
                        arm.worstAttachmentState.clear();
                        for (const auto& each : stepped.corners)
                        {
                            arm.worstAttachmentState += each.rangeLimit == RangeLimit::Droop ? "D" : each.rangeLimit == RangeLimit::Bump ? "B" : "-";
                            arm.worstAttachmentState += each.forces.droopStop != 0.0 ? "s " : "  ";
                        }
                    }
                    arm.sweeps = std::max(arm.sweeps, stepped.corners[index].frameSweeps);
                }
                const auto& solution = stepped.corners[index];
                const auto now = solution.rangeLimit != RangeLimit::None;
                arm.heldTicks += now ? 1 : 0;
                arm.arrivals += now && !held[index] ? 1 : 0;
                if (now && arm.arrivalTick[index] < 0)
                {
                    arm.arrivalTick[index] = step + 1;
                }
                held[index] = now;
                REQUIRE(solution.forces.tireVertical == 0.0);
                if (now)
                {
                    REQUIRE(solution.rangeLimit == RangeLimit::Droop);
                    REQUIRE(solution.forces.rangeLimit > 0.0);
                }
                arm.reaction[index] = solution.forces.rangeLimit;
                arm.frame[index] = solution.forces.frame;

                // What the limit has to carry with the wheel at rest on it: minus every other
                // generalised force at that position, at the wheel — the spring through its own
                // Jacobian and the stop through the shaft's, over the wheel's vertical Jacobian.
                const auto& corner = setup.corners[index];
                const auto spring = solveElement(corner.hardpoints, springElementOf(corner.hardpoints),
                                                 state.corners[index].wishboneAngle);
                const auto shaft = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints),
                                                state.corners[index].wishboneAngle);
                arm.expected[index] = -(solution.forces.spring * spring.lengthPerAngle +
                                        solution.forces.droopStop * shaft.lengthPerAngle) /
                                      solution.suspension.travelPerAngle;
            }

            // A wheel that no longer falls out under its own weight is pushed out by its spring's
            // preload alone, against the stop, so the rears arrive later than the build before's
            // fifth of a second; by one second all four hang, in both arms.
            if (step > 360)
            {
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    REQUIRE(held[index]);
                    REQUIRE(state.corners[index].wishboneRate == 0.0);
                    REQUIRE(state.corners[index].wishboneAngle == setup.corners[index].hardpoints.droopAngle);
                }
                arm.fallRateError =
                    std::max(arm.fallRateError, std::abs((state.chassis.linearVelocity.y - before) + gravity * tick));
            }
        }

        std::printf("  %s: ledger %.2e N·s, |Q/I·dt − Δq̇| %.2e, free-fall error %.2e m/s per tick, held corner-ticks %d, "
                    "arrivals %d at ticks %d %d %d %d, sweeps %u, |a assumed − a actual| %.2e m/s²\n",
                    arm.name, arm.worstResidual, arm.worstAcceleration, arm.fallRateError, arm.heldTicks, arm.arrivals,
                    arm.arrivalTick[0], arm.arrivalTick[1], arm.arrivalTick[2], arm.arrivalTick[3], arm.sweeps,
                    arm.worstAttachment);
        if (arm.frameAcceleration)
        {
            std::printf("    worst |a assumed − a actual| at tick %d, %u sweeps, corners %s\n", arm.worstAttachmentTick,
                        arm.worstAttachmentSweeps, arm.worstAttachmentState.c_str());
        }
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            std::printf("    %s: reaction at the wheel %.2f N, spring less stop %.2f N, frame term %.2f N, m_u·g %.2f N\n",
                        cornerAbbreviation(static_cast<Corner>(index)), arm.reaction[index], arm.expected[index],
                        arm.frame[index], setup.corners[index].unsprungMass * gravity);
        }

        // The true momentum's ledger closes to rounding on every tick but the four arrivals, where the
        // body's angular acceleration is tens of rad/s² for one tick and the displaced wheels' inertial
        // force is explicit in it (docs/chassis-kinematic-reaction-brief.md): second order there.
        REQUIRE(arm.worstResidual < 1e-3);
        REQUIRE(arm.worstAcceleration < 1e-9);
        REQUIRE(arm.arrivals == 4);
        REQUIRE(arm.heldTicks > 1440);
        REQUIRE(arm.fallRateError < 1e-12);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto weight = setup.corners[index].unsprungMass * gravity;
            if (arm.frameAcceleration)
            {
                // The witness: the spring less the stop, the weight given back by the frame term —
                // on the wheel's whole Jacobian, which is the body-frame vertical one to the few
                // nanoradians of pitch the arrivals' reactions gave the body.
                REQUIRE(arm.reaction[index] == Catch::Approx(arm.expected[index]).margin(1e-6));
                REQUIRE(arm.frame[index] == Catch::Approx(weight).epsilon(1e-6));
            }
            else
            {
                // The build before: the weight in the reaction, nothing in the frame term.
                REQUIRE(arm.reaction[index] == Catch::Approx(arm.expected[index] + weight).margin(1e-6));
                REQUIRE(arm.frame[index] == 0.0);
            }
        }
    }

    // The corner assumed the acceleration the chassis then had, on every tick — the arrivals, where
    // the reactions are not zero, included.
    REQUIRE(arms[0].worstAttachment < 1e-9);
    REQUIRE(arms[0].sweeps >= 1);
    REQUIRE(arms[0].sweeps <= 12);

    // And the rear's number, for the record against the brief's 629 / 207.
    REQUIRE(arms[1].reaction[2] == Catch::Approx(625.0).margin(40.0));
    const auto rearWeight = golfGtiMk7().value().corners[2].unsprungMass * gravity;
    REQUIRE(arms[0].reaction[2] == Catch::Approx(arms[1].reaction[2] - rearWeight).margin(1e-6));
}

TEST_CASE("a car in free fall from a relative equilibrium keeps its wheels where they are, at any attitude",
          "[physics][suspension][frame-acceleration]")
{
    // The unloaded Golf — springs at zero force at design, no damper, no bar, stops out of reach —
    // let go in the air with `q = 0` and `q̇ = 0`, level and at four attitudes: with the corner
    // equation right nothing moves the wheels relative to the body, because both are falling at g.
    // The build before dropped each wheel relative to the body at g over the Jacobian, which is the
    // positive control that says this fixture can tell.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    struct Attitude
    {
        const char* name;
        double pitch;
        double roll;
    };
    const auto attitudes = std::array<Attitude, 5>{Attitude{"level", 0.0, 0.0}, Attitude{"pitch +10°", 10.0, 0.0},
                                                   Attitude{"pitch -10°", -10.0, 0.0}, Attitude{"roll 10°", 0.0, 10.0},
                                                   Attitude{"pitch 6°, roll -8°", 6.0, -8.0}};

    std::printf("\n=== free fall from equilibrium, 2 s, worst |q| and |q̇| over the four corners ===\n");
    for (const auto& pose : attitudes)
    {
        auto worstAngle = std::array<double, 2>{};
        auto worstRate = std::array<double, 2>{};
        auto worstResidual = std::array<double, 2>{};
        auto worstAttachment = 0.0;
        auto worstFrame = 0.0;

        for (const auto arm : {std::size_t{0}, std::size_t{1}})
        {
            const auto setup = unloadedGolf(arm == 1);
            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, 40.0, 200.0);
            state.chassis.orientation = attitude(pose.pitch, pose.roll);
            primeMassProperties(setup, state, world.value());

            for (auto step = 0; step < 720; step++)
            {
                auto stepped = VehicleStep{};
                const auto ledger = stepLedger(setup, state, world.value(), tick, stepped);
                worstResidual[arm] = std::max(worstResidual[arm], glm::length(ledger.residual));
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    worstAngle[arm] = std::max(worstAngle[arm], std::abs(state.corners[index].wishboneAngle));
                    worstRate[arm] = std::max(worstRate[arm], std::abs(state.corners[index].wishboneRate));
                    REQUIRE(stepped.corners[index].forces.antiRoll == 0.0);
                    if (arm == 1)
                    {
                        worstAttachment = std::max(worstAttachment, ledger.attachmentError[index]);
                        // The weight, given back, at the wheel: the spring is at zero and nothing
                        // else acts, so the frame term is what the corner's generalised force is.
                        worstFrame = std::max(worstFrame, std::abs(stepped.corners[index].forces.frame -
                                                                   setup.corners[index].unsprungMass * gravity));
                    }
                }
            }
        }

        std::printf("  %-20s OFF: |q| %.3e rad |q̇| %.3e rad/s   ON: |q| %.3e rad |q̇| %.3e rad/s, ledger %.1e N·s, "
                    "|a assumed − actual| %.1e, |frame − m_u·g| %.1e N\n",
                    pose.name, worstAngle[0], worstRate[0], worstAngle[1], worstRate[1], worstResidual[1],
                    worstAttachment, worstFrame);

        // The build before moved the wheels out to their droop limits in the first tenth of a second.
        REQUIRE(worstAngle[0] > 0.5 * std::abs(unloadedGolf(false).corners[2].hardpoints.droopAngle));
        // The corrected one does not move them at all, to rounding.
        REQUIRE(worstAngle[1] < 1e-9);
        REQUIRE(worstRate[1] < 1e-8);
        REQUIRE(worstResidual[1] < 1e-9);
        REQUIRE(worstAttachment < 1e-9);
        REQUIRE(worstFrame < 1e-6);
    }

    // And with the authored stops in reach: nothing engages one, because nothing moves.
    {
        const auto setup = unloadedGolf(true, false, true);
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 40.0, 200.0);
        primeMassProperties(setup, state, world.value());
        auto stopForce = 0.0;
        for (auto step = 0; step < 720; step++)
        {
            const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());
            for (const auto& solution : stepped->corners)
            {
                stopForce = std::max({stopForce, std::abs(solution.forces.bumpStop), std::abs(solution.forces.droopStop)});
                REQUIRE(solution.rangeLimit == RangeLimit::None);
            }
        }
        std::printf("  with the authored stops in reach: largest stop force over 2 s %.3e N\n", stopForce);
        REQUIRE(stopForce == 0.0);
    }
}

// =================================================================================================
// The supported car
// =================================================================================================

TEST_CASE("on the road at rest the corrected corner is the old one: ride, loads and balance to rounding",
          "[physics][suspension][frame-acceleration]")
{
    // The distinction the term rests on: a supported chassis does not accelerate, so the term reads
    // zero and the wheel's weight is carried as it always was — by the tyre through the corner. The
    // Golf placed on the plate and settled for four seconds, in both arms, compared corner by corner.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    // The shipped dampers carry seal friction, so the car comes to rest anywhere inside a band a
    // few micrometres wide and the two arms' transients end at different places in it; the
    // friction-free car has one rest position. Even there the two arms differ by a few hundredths
    // of a newton at each corner, and that is the weight's projection: the build before carried the
    // unsprung weight on the body-frame vertical Jacobian, the corrected corner on the world one,
    // and the car at rest sits at a small pitch and roll. The chassis's acceleration itself reads
    // zero to rounding, and every difference is bounded by that projection.
    for (const auto friction : {false, true})
    {
    auto angle = std::array<std::array<double, cornerCount>, 2>{};
    auto load = std::array<std::array<double, cornerCount>, 2>{};
    auto spring = std::array<std::array<double, cornerCount>, 2>{};
    auto frame = std::array<double, 2>{};
    auto attachment = std::array<double, 2>{};
    auto projection = 0.0; // the largest weight-projection difference, N at the wheel
    auto pitch = 0.0;
    auto roll = 0.0;
    auto weight = 0.0;

    for (const auto arm : {std::size_t{0}, std::size_t{1}})
    {
        auto setup = golfInVacuum(arm == 1);
        if (!friction)
        {
            for (auto& corner : setup.corners)
            {
                corner.damperFriction = 0.0;
            }
        }
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 1.0, 200.0);
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), 1e-6).has_value());
        const auto centreOfMass = state.chassis.centreOfMass;
        state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, centreOfMass.y - 0.008, 200.0);

        auto stepped = VehicleStep{};
        for (auto step = 0; step < 1440; step++)
        {
            const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(result.has_value());
            stepped = result.value();
        }
        weight = state.chassis.mass * gravity;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            angle[arm][index] = state.corners[index].wishboneAngle;
            load[arm][index] = stepped.corners[index].forces.tireVertical;
            spring[arm][index] = stepped.corners[index].forces.spring;
            frame[arm] = std::max(frame[arm], std::abs(stepped.corners[index].forces.frame));
            attachment[arm] = std::max(attachment[arm], glm::length(stepped.corners[index].attachmentAcceleration));

            const auto& suspension = stepped.corners[index].suspension;
            const auto jacobian = state.chassis.orientation * suspension.wheelCentrePerAngle;
            projection = std::max(projection, std::abs(setup.corners[index].unsprungMass * gravity *
                                                       (jacobian.y - suspension.travelPerAngle) / suspension.travelPerAngle));
        }
        const auto forward = state.chassis.orientation * glm::dvec3(0.0, 0.0, 1.0);
        const auto right = state.chassis.orientation * glm::dvec3(1.0, 0.0, 0.0);
        pitch = std::asin(forward.y) * degrees;
        roll = -std::asin(right.y) * degrees;
    }

    std::printf("\n=== the Golf at rest on the plate, after 4 s, seal friction %s ===\n", friction ? "as shipped" : "off");
    auto total = std::array<double, 2>{};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        std::printf("  %s: angle OFF %.9f ON %.9f rad, Fz OFF %.6f ON %.6f N, spring OFF %.6f ON %.6f N\n",
                    cornerAbbreviation(static_cast<Corner>(index)), angle[0][index], angle[1][index], load[0][index],
                    load[1][index], spring[0][index], spring[1][index]);
        total[0] += load[0][index];
        total[1] += load[1][index];
        const auto band = friction ? 2e-5 : 2.0 * projection / 20000.0; // rad: the projection over a 20 kN/m wheel rate, doubled
        REQUIRE(angle[1][index] == Catch::Approx(angle[0][index]).margin(band));
        REQUIRE(load[1][index] == Catch::Approx(load[0][index]).margin(friction ? 0.5 : 2.0 * projection + 1e-6));
        REQUIRE(spring[1][index] == Catch::Approx(spring[0][index]).margin(friction ? 0.5 : 2.0 * projection + 1e-6));
    }
    std::printf("  ΣFz OFF %.4f ON %.4f against W %.4f N; body pitch %.4f° roll %.4f°; ON: |frame| %.2e N, weight projection %.2e N, |a attach| %.2e m/s²\n",
                total[0], total[1], weight, pitch, roll, frame[1], projection, attachment[1]);
    REQUIRE(total[1] == Catch::Approx(weight).margin(0.05));
    REQUIRE(frame[1] <= projection + 0.01);
    REQUIRE(attachment[1] < 1e-4);
    REQUIRE(frame[0] == 0.0);
    }
}

// =================================================================================================
// Base acceleration: the modes of the airborne car
// =================================================================================================

TEST_CASE("the wheels of the airborne car oscillate at the reduced-mass frequency, not the fixed-chassis one",
          "[physics][suspension][frame-acceleration]")
{
    // The controlled base-acceleration case, with an analytical reference. The unloaded Golf in the
    // air with no damper is, to first order, four springs between four unsprung masses and one free
    // rigid body: `K·x = ω²·A·x` with `A = diag(m_u·|C'|²) − m_u·G·m_u`, the chassis mobility taking
    // the place of an infinitely heavy body. The build before is the same problem with `A` diagonal.
    // Each mode is excited on its own shape, in both arms, and the period read off its zero
    // crossings. Which reference each arm follows is what this proves; that the two references
    // differ by several per cent is what makes it a proof rather than a tautology.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    std::printf("\n=== the airborne car's modes, 2 s at 360 Hz ===\n");
    for (const auto arm : {std::size_t{1}, std::size_t{0}})
    {
        const auto setup = unloadedGolf(arm == 1);
        auto primed = VehicleState{};
        primed.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
        primeMassProperties(setup, primed, world.value());

        const auto coupled = linearise(setup, primed, true);
        const auto fixed = linearise(setup, primed, false);
        const auto coupledModes = solveModes(coupled.inertia, coupled.stiffness);
        const auto fixedModes = solveModes(fixed.inertia, fixed.stiffness);
        const auto& reference = arm == 1 ? coupledModes : fixedModes;
        const auto& other = arm == 1 ? fixedModes : coupledModes;

        std::printf("  %s: reference modes", arm == 1 ? "ON " : "OFF");
        for (auto mode = std::size_t{0}; mode < cornerCount; mode++)
        {
            std::printf("  %.4f Hz (shape %+.2f %+.2f %+.2f %+.2f)", reference.omega[mode] / (2.0 * 3.14159265358979323846),
                        reference.x[mode][0], reference.x[mode][1], reference.x[mode][2], reference.x[mode][3]);
        }
        std::printf("\n");

        for (auto mode = std::size_t{0}; mode < cornerCount; mode++)
        {
            auto state = primed;
            auto largest = 0.0;
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                largest = std::max(largest, std::abs(reference.x[mode][index]));
            }
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                state.corners[index].wishboneAngle = 0.01 * reference.x[mode][index] / largest;
            }

            // The mode's own coordinate: its shape against the arm's mass matrix on the rates.
            auto record = std::vector<double>{};
            const auto& massMatrix = arm == 1 ? coupled.inertia : fixed.inertia;
            for (auto step = 0; step < 720; step++)
            {
                REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
                auto projected = 0.0;
                for (auto i = std::size_t{0}; i < cornerCount; i++)
                {
                    for (auto j = std::size_t{0}; j < cornerCount; j++)
                    {
                        projected += reference.x[mode][i] * massMatrix[i][j] * state.corners[j].wishboneRate;
                    }
                }
                record.push_back(projected);
            }

            const auto period = periodOf(record, tick);
            const auto measured = 2.0 * 3.14159265358979323846 / period;
            const auto againstReference = measured / reference.omega[mode] - 1.0;
            const auto againstOther = measured / other.omega[mode] - 1.0;
            std::printf("    mode %zu: measured %.4f Hz, against its reference %+.3f %%, against the other arm's %+.3f %%\n",
                        mode, measured / (2.0 * 3.14159265358979323846), 100.0 * againstReference, 100.0 * againstOther);

            // The build before sags its wheels under their own weight to a new equilibrium tens of
            // milliradians out, where the strut's Jacobian is a couple of per cent different, so it
            // is held to its reference loosely; the corrected car oscillates about q = 0.
            REQUIRE(std::abs(againstReference) < (arm == 1 ? 0.004 : 0.025));
            REQUIRE(std::abs(againstOther) > 0.01);
        }
    }
}

// =================================================================================================
// Pitch and roll: the attachment's own acceleration
// =================================================================================================

TEST_CASE("a rolling body flings its hanging wheels outward, both sides alike, and a pitching one lifts the end it lifts",
          "[physics][suspension][frame-acceleration]")
{
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    // The Golf in the air with all four wheels hanging from their limits, so that the frame term
    // reads straight into the constraint's reaction and nothing moves.
    const auto hang = [&](const VehicleSetup& setup, VehicleState& state)
    {
        state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
        for (auto step = 0; step < 180; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            REQUIRE(state.corners[index].wishboneAngle == setup.corners[index].hardpoints.droopAngle);
        }
    };

    SECTION("a roll rate: the centripetal term")
    {
        const auto setup = golfInVacuum(true);
        auto state = VehicleState{};
        hang(setup, state);

        const auto still = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(still.has_value());

        // Three radians a second about the body's forward axis.
        setAngularVelocity(state.chassis, state.chassis.orientation * glm::dvec3(0.0, 0.0, 3.0));
        const auto before = state;
        const auto rolling = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(rolling.has_value());

        std::printf("\n=== a hanging car rolling at 3 rad/s: what the limit carries, still -> rolling ===\n");
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& quiet = still->corners[index];
            const auto& spun = rolling->corners[index];
            REQUIRE(quiet.rangeLimit == RangeLimit::Droop);
            REQUIRE(spun.rangeLimit == RangeLimit::Droop);

            // The independent statement: the body-fixed point under the wheel centre accelerates
            // toward the roll axis by ω² times its distance from it; projected onto the wheel's
            // Jacobian and given back through the unsprung mass, at the wheel.
            // — plus the angular acceleration the body actually had this tick: the gyroscopic one a
            // turning body gives itself and, since 2026-09-08 latest of all (d), what the displaced
            // hanging wheels' own inertial forces and couples do to it.
            const auto omega = angularVelocity(before.chassis);
            const auto gyroscopic = (angularVelocity(state.chassis) - omega) / tick;
            const auto arm = bodyToWorld(before.chassis, spun.suspension.wheelCentre) - before.chassis.position;
            // And the body's own linear acceleration beyond free fall, which the hanging wheels'
            // centripetal reactions give it (docs/chassis-kinematic-reaction-brief.md).
            const auto beyondFall = (state.chassis.linearVelocity - before.chassis.linearVelocity) / tick - glm::dvec3(0.0, -gravity, 0.0);
            const auto centripetal = glm::cross(omega, glm::cross(omega, arm)) + glm::cross(gyroscopic, arm) + beyondFall;
            const auto jacobian = before.chassis.orientation * spun.suspension.wheelCentrePerAngle;
            const auto expected = -setup.corners[index].unsprungMass * glm::dot(jacobian, centripetal) /
                                  spun.suspension.travelPerAngle;

            const auto frameChange = spun.forces.frame - quiet.forces.frame;
            const auto reactionChange = spun.forces.rangeLimit - quiet.forces.rangeLimit;
            std::printf("  %s: frame %+.3f -> %+.3f N (change %+.3f, the turning body says %+.3f, gyroscopic α (%.3f, %.3f, %.3f) rad/s²), the limit %+.3f -> %+.3f N\n",
                        cornerAbbreviation(static_cast<Corner>(index)), quiet.forces.frame, spun.forces.frame,
                        frameChange, expected, gyroscopic.x, gyroscopic.y, gyroscopic.z, quiet.forces.rangeLimit,
                        spun.forces.rangeLimit);

            REQUIRE(frameChange == Catch::Approx(expected).margin(1e-3));
            // Flung outward is toward droop for a wheel below the axis: the limit pulls harder.
            REQUIRE(frameChange < 0.0);
            REQUIRE(reactionChange == Catch::Approx(-frameChange).margin(1e-6));
        }
        // Both sides alike.
        REQUIRE(rolling->corners[0].forces.frame == Catch::Approx(rolling->corners[1].forces.frame).margin(1e-6));
        REQUIRE(rolling->corners[2].forces.frame == Catch::Approx(rolling->corners[3].forces.frame).margin(1e-6));
    }

    SECTION("a pitch acceleration: lift ahead of the centre of mass")
    {
        auto setup = golfInVacuum(true);
        auto state = VehicleState{};
        hang(setup, state);

        // Flying forward at 40 m/s with a wing 1.5 m ahead of the centre of mass and no drag: a
        // known lift, a known pitch torque, and the front's attachment accelerating up while the
        // rear's goes down.
        setup.aero.push_back(AeroSurface{.centre = state.chassis.centreOfMass + glm::dvec3(0.0, 0.0, 1.5),
                                         .dragArea = 0.0,
                                         .liftArea = 2.0});
        state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, 40.0);

        const auto before = state;
        auto stepped = VehicleStep{};
        const auto ledger = stepLedger(setup, state, world.value(), tick, stepped);

        const auto lift = 0.5 * setup.airDensity * 1600.0 * 2.0;
        const auto linear = (state.chassis.linearVelocity - before.chassis.linearVelocity) / tick;
        const auto alpha = (angularVelocity(state.chassis) - angularVelocity(before.chassis)) / tick;

        std::printf("\n=== a hanging car with %.0f N of lift 1.5 m ahead: chassis a_y %.3f m/s² (lift/M − g = %.3f), α_x %.3f rad/s²\n",
                    lift, linear.y, lift / state.chassis.mass - gravity, alpha.x);
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = stepped.corners[index];
            std::printf("  %s: attachment a_y assumed %.4f m/s², |assumed − actual| %.2e, frame %+.2f N, limit %+.2f N\n",
                        cornerAbbreviation(static_cast<Corner>(index)), solution.attachmentAcceleration.y,
                        ledger.attachmentError[index], solution.forces.frame, solution.forces.rangeLimit);
            REQUIRE(ledger.attachmentError[index] < 1e-6);
        }

        // Front up, rear down, relative to the lift alone; both sides alike.
        const auto& fl = stepped.corners[0];
        const auto& fr = stepped.corners[1];
        const auto& rl = stepped.corners[2];
        const auto& rr = stepped.corners[3];
        REQUIRE(fl.attachmentAcceleration.y > linear.y);
        REQUIRE(rl.attachmentAcceleration.y < linear.y);
        REQUIRE(fl.forces.frame < rl.forces.frame);
        REQUIRE(fl.attachmentAcceleration.y == Catch::Approx(fr.attachmentAcceleration.y).margin(1e-9));
        REQUIRE(rl.attachmentAcceleration.y == Catch::Approx(rr.attachmentAcceleration.y).margin(1e-9));
        // The wing is an external impulse the ledger does not carry: the residual is it, to the
        // second order of the hanging wheels' explicit inertial force under the body's pitch.
        REQUIRE(glm::length(ledger.residual - glm::dvec3(0.0, lift * tick, 0.0)) < 1e-5);
    }
}

// =================================================================================================
// Momentum and energy
// =================================================================================================

TEST_CASE("the airborne car's energy in the falling frame is kept without a damper and paid out through one",
          "[physics][suspension][frame-acceleration]")
{
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    std::printf("\n=== energy in the falling frame, 2 s at 360 Hz ===\n");
    for (const auto damped : {false, true})
    {
        for (const auto arm : {std::size_t{0}, std::size_t{1}})
        {
            const auto setup = unloadedGolf(arm == 1, damped);
            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
            primeMassProperties(setup, state, world.value());

            // The heave-like lowest coupled mode's shape, 10 mrad at the largest corner, and a
            // little roll rate so the cross term is exercised.
            const auto coupled = linearise(setup, state, true);
            const auto modes = solveModes(coupled.inertia, coupled.stiffness);
            auto largest = 0.0;
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                largest = std::max(largest, std::abs(modes.x[0][index]));
            }
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                state.corners[index].wishboneAngle = 0.01 * modes.x[0][index] / largest;
            }
            setAngularVelocity(state.chassis, glm::dvec3(0.0, 0.0, 0.2));

            const auto velocityAtStart = state.chassis.linearVelocity;
            auto stepped = VehicleStep{};
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), 1e-12).has_value());
            const auto initial = fallingFrameEnergy(setup, state, stepped, velocityAtStart, 0.0);

            auto worstResidual = 0.0;
            auto largestDrift = 0.0;
            auto damperWork = 0.0;
            auto energy = initial;
            for (auto step = 0; step < 720; step++)
            {
                const auto ledger = stepLedger(setup, state, world.value(), tick, stepped);
                worstResidual = std::max(worstResidual, glm::length(ledger.residual));
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    // The damper's force on the shaft times the shaft's compression rate at the rate
                    // that acted: what it took out this tick.
                    const auto shaft = solveElement(setup.corners[index].hardpoints,
                                                    damperElementOf(setup.corners[index].hardpoints),
                                                    state.corners[index].wishboneAngle);
                    damperWork += stepped.corners[index].forces.damper *
                                  (-shaft.lengthPerAngle * state.corners[index].wishboneRate) * tick;
                }
                energy = fallingFrameEnergy(setup, state, stepped, velocityAtStart,
                                            static_cast<double>(step + 1) * tick);
                largestDrift = std::max(largestDrift, std::abs(energy - initial - (damped ? -damperWork : 0.0)));
            }

            std::printf("  %s %s: initial %.6f J, final %.6f J, damper work %.6f J, worst |ΔE%s| %.3e J, ledger %.1e N·s\n",
                        damped ? "damped  " : "springs ", arm == 1 ? "ON " : "OFF", initial, energy, damperWork,
                        damped ? " + W_d" : "", largestDrift, worstResidual);

            // The wheels move here, so the chassis's explicit centripetal reaction leaves the true
            // ledger a second-order residual per tick (docs/chassis-kinematic-reaction-brief.md).
            REQUIRE(worstResidual < 1e-2);
            if (arm == 1)
            {
                // Bounded, and small against the mode's own energy: the semi-implicit step's
                // oscillating first-order error, with nothing accumulating.
                REQUIRE(largestDrift < 0.02 * initial);
                if (damped)
                {
                    REQUIRE(energy < initial);
                    REQUIRE(damperWork > 0.0);
                    REQUIRE(initial - energy == Catch::Approx(damperWork).epsilon(0.02));
                }
            }
        }
    }
}

// =================================================================================================
// Timestep robustness
// =================================================================================================

TEST_CASE("the corrected corner converges with the timestep: no relative drift in free fall, the mode's frequency to second order",
          "[physics][suspension][frame-acceleration]")
{
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    std::printf("\n=== timestep sweep, 2 s each ===\n");
    auto lastFrequencyError = 1e300;
    for (const auto hertz : {120.0, 360.0, 720.0, 2880.0})
    {
        const auto dt = 1.0 / hertz;
        const auto ticks = static_cast<int>(std::lround(2.0 * hertz));
        const auto setup = unloadedGolf(true);

        // Free fall from equilibrium.
        auto drift = 0.0;
        auto worstResidual = 0.0;
        auto worstAttachment = 0.0;
        {
            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
            primeMassProperties(setup, state, world.value());
            for (auto step = 0; step < ticks; step++)
            {
                auto stepped = VehicleStep{};
                const auto ledger = stepLedger(setup, state, world.value(), dt, stepped);
                worstResidual = std::max(worstResidual, glm::length(ledger.residual));
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    drift = std::max(drift, std::abs(state.corners[index].wishboneAngle));
                    worstAttachment = std::max(worstAttachment, ledger.attachmentError[index]);
                }
            }
        }

        // The lowest mode.
        auto frequencyError = 0.0;
        auto energyDrift = 0.0;
        {
            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
            primeMassProperties(setup, state, world.value());
            const auto coupled = linearise(setup, state, true);
            const auto modes = solveModes(coupled.inertia, coupled.stiffness);
            auto largest = 0.0;
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                largest = std::max(largest, std::abs(modes.x[0][index]));
            }
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                state.corners[index].wishboneAngle = 0.01 * modes.x[0][index] / largest;
            }
            const auto velocityAtStart = state.chassis.linearVelocity;
            auto stepped = VehicleStep{};
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), 1e-12).has_value());
            const auto initial = fallingFrameEnergy(setup, state, stepped, velocityAtStart, 0.0);

            auto record = std::vector<double>{};
            for (auto step = 0; step < ticks; step++)
            {
                REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), dt).has_value());
                auto projected = 0.0;
                for (auto i = std::size_t{0}; i < cornerCount; i++)
                {
                    for (auto j = std::size_t{0}; j < cornerCount; j++)
                    {
                        projected += modes.x[0][i] * coupled.inertia[i][j] * state.corners[j].wishboneRate;
                    }
                }
                record.push_back(projected);
            }
            const auto measured = 2.0 * 3.14159265358979323846 / periodOf(record, dt);
            frequencyError = std::abs(measured / modes.omega[0] - 1.0);
            const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), 1e-12);
            REQUIRE(result.has_value());
            energyDrift = std::abs(fallingFrameEnergy(setup, state, result.value(), velocityAtStart, 2.0) - initial) / initial;
        }

        std::printf("  %6.0f Hz: free-fall drift %.2e rad, ledger %.1e N·s, |a assumed − actual| %.1e m/s², mode frequency error %.4f %%, "
                    "energy drift %.3f %%\n",
                    hertz, drift, worstResidual, worstAttachment, 100.0 * frequencyError, 100.0 * energyDrift);

        REQUIRE(drift < 1e-9);
        REQUIRE(worstResidual < 1e-9);
        REQUIRE(worstAttachment < 1e-9);
        REQUIRE(frequencyError <= lastFrequencyError + 1e-5);
        REQUIRE(frequencyError < 0.005);
        lastFrequencyError = frequencyError;
    }
}

// =================================================================================================
// The closed airborne car with the corner's configuration-dependent inertia term
// =================================================================================================

TEST_CASE("the closed airborne car with the geometry term: momentum still closes, the falling-frame energy is kept at least as well, OLD against CORRECTED",
          "[physics][suspension][frame-acceleration][nonlinear-geometry]")
{
    // The unloaded Golf on its lowest coupled mode at 60 mrad — six times the frame proofs'
    // amplitude, rates to 1.5 rad/s, where the term is a few tenths of a per cent of the spring's
    // generalised force — with a 0.2 rad/s roll rate, springs only, 2 s, `nonlinearGeometry` off and
    // on (docs/nonlinear-geometry-brief.md). The term is a generalised force like any other on the
    // corner's side, so the chassis reaction reads it through `Q/I` and the ledger must still close;
    // and it is even in the rate, so over a symmetric cycle it does no net work and the falling-frame
    // energy's ripple must not grow.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    std::printf("\n=== the closed airborne car, lowest mode at 60 mrad, 2 s at 360 Hz ===\n");
    auto drift = std::array<double, 2>{};
    for (const auto arm : {std::size_t{0}, std::size_t{1}})
    {
        auto setup = unloadedGolf(true, false);
        setup.nonlinearGeometry = arm == 1;
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 60.0, 200.0);
        primeMassProperties(setup, state, world.value());

        const auto coupled = linearise(setup, state, true);
        const auto modes = solveModes(coupled.inertia, coupled.stiffness);
        auto largest = 0.0;
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            largest = std::max(largest, std::abs(modes.x[0][index]));
        }
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            state.corners[index].wishboneAngle = 0.06 * modes.x[0][index] / largest;
        }
        setAngularVelocity(state.chassis, glm::dvec3(0.0, 0.0, 0.2));

        const auto velocityAtStart = state.chassis.linearVelocity;
        auto stepped = VehicleStep{};
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), 1e-12).has_value());
        const auto initial = fallingFrameEnergy(setup, state, stepped, velocityAtStart, 0.0);

        auto worstResidual = 0.0;
        auto worstAcceleration = 0.0;
        auto worstAttachment = 0.0;
        auto largestTerm = 0.0;
        auto largestRate = 0.0;
        auto energy = initial;
        for (auto step = 0; step < 720; step++)
        {
            const auto ledger = stepLedger(setup, state, world.value(), tick, stepped);
            worstResidual = std::max(worstResidual, glm::length(ledger.residual));
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                worstAcceleration = std::max(worstAcceleration, ledger.accelerationError[index]);
                worstAttachment = std::max(worstAttachment, ledger.attachmentError[index]);
                largestTerm = std::max(largestTerm, std::abs(stepped.corners[index].forces.geometry));
                largestRate = std::max(largestRate, std::abs(state.corners[index].wishboneRate));
                if (arm == 0)
                {
                    REQUIRE(stepped.corners[index].forces.geometry == 0.0);
                    REQUIRE(stepped.corners[index].inertiaSlope == 0.0);
                }
            }
            energy = fallingFrameEnergy(setup, state, stepped, velocityAtStart, static_cast<double>(step + 1) * tick);
            drift[arm] = std::max(drift[arm], std::abs(energy - initial));
        }

        std::printf("  %s: initial %.5f J, final %.5f J, worst |ΔE| %.4e J (%.3f %%), ledger %.1e N·s, |Q/I·dt − Δq̇| %.1e, "
                    "|a assumed − actual| %.1e, |q̇| max %.3f rad/s, |geometry| max %.3f N at the wheel\n",
                    arm == 1 ? "CORRECTED" : "OLD      ", initial, energy, drift[arm], 100.0 * drift[arm] / initial,
                    worstResidual, worstAcceleration, worstAttachment, largestRate, largestTerm);

        REQUIRE(worstResidual < 2e-2);
        REQUIRE(worstAcceleration < 1e-9);
        // The body rolls at 0.2 rad/s under reactions six times the frame proofs', so the tensor's
        // turn within the tick — second order, which the prediction's Euler term is not — reads a
        // few micrometres per second squared here.
        REQUIRE(worstAttachment < 1e-4);
        REQUIRE(drift[arm] < 0.05 * initial);
    }
    REQUIRE(drift[1] <= drift[0] * 1.05);
}
