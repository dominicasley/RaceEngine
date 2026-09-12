// The linkage's authored range as a unilateral constraint (docs/range-constraint-brief.md, 2026-09-08
// later still): the proofs.
//
// Until this change the range `[droopAngle, bumpAngle]` was a clamp on the state in pass three —
// the coordinate snapped onto the limit and the rate floored — applied after pass two had given the
// chassis the reaction of the *unconstrained* acceleration and after the damper's and the stop's
// implicit shares had been read off the rate the clamp then threw away. The droop audit measured
// three consequences on the rear droop limit: the body pushed up by a wheel that was hanging from
// it (the four tyre loads summing to the weight minus the clamped deficits), a damper force of
// hundreds of newtons read on a stationary shaft, and a corner that chattered on the limit and, with
// enough mass behind the axle, a car that pitched onto its nose and left the world.
//
// The constraint now lives in pass two (`constrainCornerRange`): the velocity step's rate is kept
// whenever it lands inside the range and replaced by the one rate that lands exactly on the limit
// when it does not; the implicit shares are read off the rate that acts; and the constraint's
// generalised force — the residual between the corner's discrete acceleration and every other force
// at that rate — is folded into the generalised force the chassis reaction reads, so the body gets
// the equal and opposite reaction through the same Jacobian as every other corner force.
//
// Two levels of proof. The element cases run the force pass's velocity-step block in shaft
// coordinates — the production functions `solveCornerRate`, `constrainCornerRange`, the stop's
// `terms` and the damper's `solveDamperForce`/`damperDampingCoefficient` in the production order —
// on one corner reduced to its damper shaft, with the clamp of the build before written out beside
// it as the control. The whole-car cases run `stepVehicle` itself: the Golf in the air, where every
// corner hangs from its droop limit and the chassis is a free body, and the Golf on the plate with a
// mass behind an axle, where a corner is held against its limit by the road.
//
// Shaft coordinates here: `x` is the shaft's compression, positive in bump, `v = ẋ`; the corner's
// coordinate is `q = x` through a damper Jacobian of −1 (a real damper shortens as the wishbone
// rises), so a force that pushes the shaft LONGER — the spring's, the damper's under compression,
// the bump stop's — is a negative generalised force, exactly as on the car. Every table prints;
// `./EngineTests "[range-constraint]" -s` shows them.

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

using raceengine::bringUpJolt;
using raceengine::constrainCornerRange;
using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::CornerRateStep;
using raceengine::CornerSetup;
using raceengine::Curve;
using raceengine::damperDampingCoefficient;
using raceengine::damperElementOf;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::MassComponent;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::RangeConstraintStep;
using raceengine::RangeLimit;
using raceengine::seedTyreGasPressures;
using raceengine::solveCornerRate;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperForce;
using raceengine::solveElement;
using raceengine::stepVehicle;
using raceengine::SuspensionState;
using raceengine::tearDownJolt;
using raceengine::TravelStop;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto gravity = 9.80665;
constexpr auto tick = 1.0 / 360.0;
constexpr auto degrees = 180.0 / 3.14159265358979323846;
// The Golf's axle stations, chassis frame (PublishedCars.cppm).
constexpr auto frontAxle = 1.319;

// --- one corner on its shaft --------------------------------------------------------------------

struct Shaft
{
    double mass = 0.0;        // the generalised inertia brought onto the shaft, kg
    double jacobian = 0.0;    // the damper element's dLength/dq at design, m/rad (negative)
    double springRate = 0.0;  // N/m on the shaft
    double preload = 0.0;     // the spring's force at design, N — what a real spring still pushes with at full droop
    double extension = 0.0;   // the shaft's extension at `droopAngle`, m — the droop limit
    double compression = 0.0; // the shaft's compression at `bumpAngle`, m — the bump limit
    SuspensionState design;
    CornerSetup corner;
};

[[nodiscard]] Shaft shaftOf(const CornerSetup& corner)
{
    const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
    REQUIRE(design.has_value());

    const auto element = damperElementOf(corner.hardpoints);
    const auto designLength = solveElement(corner.hardpoints, element, 0.0).length;
    const auto jacobian = solveElement(corner.hardpoints, element, 0.0).lengthPerAngle;
    const auto inertia = corner.unsprungMass * glm::dot(design->wheelCentrePerAngle, design->wheelCentrePerAngle);

    return Shaft{.mass = inertia / (jacobian * jacobian),
                 .jacobian = jacobian,
                 .springRate = corner.springRate,
                 .preload = raceengine::solveSpringForce(corner, *design).force,
                 .extension =
                     solveElement(corner.hardpoints, element, corner.hardpoints.droopAngle).length - designLength,
                 .compression =
                     designLength - solveElement(corner.hardpoints, element, corner.hardpoints.bumpAngle).length,
                 .design = *design,
                 .corner = corner};
}

// The production damper at a shaft velocity: its force (positive pushes the shaft longer) and its
// floored slope, both on the shaft, through the force pass's own two calls.
struct Element
{
    double force = 0.0;
    double slope = 0.0;
};

[[nodiscard]] Element damperAt(const Shaft& shaft, const double velocity)
{
    const auto damper = solveDamperForce(shaft.corner, shaft.design, -velocity / shaft.jacobian);
    return Element{.force = damper.force,
                   .slope = damperDampingCoefficient(shaft.corner, damper) / (shaft.jacobian * shaft.jacobian)};
}

struct Scenario
{
    bool spring = false;
    bool damper = false;
    bool stops = false;
    double load = 0.0; // a constant force on the shaft, newtons, positive pushing it LONGER (toward droop)
};

enum class Scheme
{
    Constraint, // the shipped block: the range decided before the shares, the reaction folded in
    Clamp       // the build before: shares at the free rate, the state clamped afterwards, no reaction
};

struct Tick
{
    double x = 0.0;        // compression after the step, m
    double v = 0.0;        // compression velocity after the step, m/s
    double free = 0.0;     // the velocity the step solved before the range was consulted
    double damper = 0.0;   // the damper force that acted, N, positive pushing longer
    double bump = 0.0;     // the bump stop's force that acted, N
    double droop = 0.0;    // the droop stop's, N, positive pushing longer... i.e. negative on the shaft
    double reaction = 0.0; // the constraint's force on the shaft, N, positive pushing SHORTER (toward bump)
    double impulse = 0.0;  // the same over the tick, N·s
    double q = 0.0;        // the generalised force the chassis reaction reads (with unit |Jacobian|), N
    RangeLimit limit = RangeLimit::None;
};

// The force pass's velocity-step block, on the shaft. `Scheme::Constraint` is the shipped block
// call for call: the free rate, the stop's push-only re-solve, the range, the shares at the rate
// that acts, the residual folded in. `Scheme::Clamp` is the build before: the shares at the free
// rate, the chassis handed the free acceleration, and then the state clamped.
[[nodiscard]] Tick advance(const Shaft& shaft, const Scenario& scenario, const Scheme scheme, const double dt,
                           const double x, const double v)
{
    const auto element = scenario.damper ? damperAt(shaft, v) : Element{};
    const auto bumpTerms = scenario.stops ? shaft.corner.bumpStop.terms(x - shaft.corner.bumpStop.gap, v)
                                          : raceengine::TravelStopTerms{};
    const auto droopTerms = scenario.stops ? shaft.corner.droopStop.terms(-x - shaft.corner.droopStop.gap, -v)
                                           : raceengine::TravelStopTerms{};

    // On the shaft, positive pushes it longer; the corner's generalised force is its negative.
    const auto spring = scenario.spring ? shaft.preload + shaft.springRate * x : 0.0;
    auto axis = spring + element.force + bumpTerms.explicitForce - droopTerms.explicitForce + scenario.load;
    auto generalised = -axis;

    const auto damperCoefficient = element.slope;
    const auto stopCoefficient = bumpTerms.viscousCoefficient + droopTerms.viscousCoefficient;

    auto rate = solveCornerRate(CornerRateStep{.previousRate = v,
                                               .generalisedForce = generalised,
                                               .generalisedInertia = shaft.mass,
                                               .damperCoefficient = damperCoefficient,
                                               .stopCoefficient = stopCoefficient,
                                               .deltaTime = dt});

    auto result = Tick{};
    result.bump = bumpTerms.explicitForce;
    result.droop = -droopTerms.explicitForce;

    const auto bump = bumpTerms.viscousCoefficient > 0.0;
    const auto& engaged = bump ? bumpTerms : droopTerms;
    auto viscous = 0.0;
    auto stopHeld = false;

    if (stopCoefficient > 0.0)
    {
        viscous = engaged.viscousCoefficient * (bump ? rate : -rate);
        if (engaged.explicitForce + viscous < 0.0)
        {
            const auto axisExplicit = bump ? engaged.explicitForce : -engaged.explicitForce;
            generalised += axisExplicit;
            rate = solveCornerRate(CornerRateStep{.previousRate = v,
                                                  .generalisedForce = generalised,
                                                  .generalisedInertia = shaft.mass,
                                                  .damperCoefficient = damperCoefficient,
                                                  .stopCoefficient = 0.0,
                                                  .deltaTime = dt});
            if (bump)
            {
                result.bump = 0.0;
            }
            else
            {
                result.droop = 0.0;
            }
        }
        else
        {
            stopHeld = true;
        }
    }

    result.free = rate;

    if (scheme == Scheme::Constraint)
    {
        const auto constrained = constrainCornerRange(RangeConstraintStep{.previousAngle = x,
                                                                          .rate = rate,
                                                                          .droopAngle = -shaft.extension,
                                                                          .bumpAngle = shaft.compression,
                                                                          .deltaTime = dt});
        if (constrained.limit != RangeLimit::None)
        {
            rate = constrained.rate;
            if (stopHeld)
            {
                viscous = engaged.viscousCoefficient * (bump ? rate : -rate);
            }
        }
        result.limit = constrained.limit;
    }

    if (stopHeld)
    {
        const auto axisViscous = bump ? viscous : -viscous;
        generalised -= axisViscous;
        if (bump)
        {
            result.bump += axisViscous;
        }
        else
        {
            result.droop += axisViscous;
        }
    }

    result.damper = element.force;
    if (damperCoefficient > 0.0)
    {
        const auto share = -damperCoefficient * (rate - v);
        generalised += share;
        result.damper -= share; // the shaft's own share: `share / jacobian` with the unit-magnitude Jacobian of −1
    }

    if (scheme == Scheme::Constraint)
    {
        if (result.limit != RangeLimit::None)
        {
            const auto reaction = shaft.mass * (rate - v) / dt - generalised;
            generalised += reaction;
            result.reaction = reaction;
            result.impulse = reaction * dt;
        }

        result.q = generalised;
        result.v = rate;
        result.x = result.limit == RangeLimit::Bump    ? shaft.compression
                   : result.limit == RangeLimit::Droop ? -shaft.extension
                                                       : x + rate * dt;
    }
    else
    {
        // The build before: the chassis reads the free acceleration, then the state is clamped.
        result.q = generalised;
        result.v = rate;
        result.x = x + rate * dt;
        if (result.x > shaft.compression)
        {
            result.x = shaft.compression;
            result.v = std::min(result.v, 0.0);
            result.limit = RangeLimit::Bump;
        }
        else if (result.x < -shaft.extension)
        {
            result.x = -shaft.extension;
            result.v = std::max(result.v, 0.0);
            result.limit = RangeLimit::Droop;
        }
    }

    return result;
}

// A run of the shaft, with the ledger the proofs read.
struct Run
{
    std::vector<Tick> ticks;
    double impulse = 0.0;        // Σ constraint impulse, N·s (positive toward bump)
    double work = 0.0;           // Σ Λ · ½ (v₀ + v₁): what the constraint did to the shaft's kinetic energy, J, exactly
    double kineticIn = 0.0;
    double kineticOut = 0.0;
    double boundaryError = 0.0;  // max overshoot past either limit, m
    double heldVelocity = 0.0;   // max |v| on a tick held from one already held (the arrival lands at its own rate)
    int heldTicks = 0;
    int contacts = 0;            // entries into either limit
    int firstHeld = -1;
    int firstFree = -1;          // the first tick after `firstHeld` with no limit
    double reactionMin = 1e300;
    double reactionMax = -1e300;
};

[[nodiscard]] Run run(const Shaft& shaft, const Scenario& scenario, const Scheme scheme, const double dt,
                      const int ticks, const double x0, const double v0,
                      const double loadAfter = 0.0, const int loadAfterTick = -1)
{
    auto result = Run{};
    result.kineticIn = 0.5 * shaft.mass * v0 * v0;
    auto x = x0;
    auto v = v0;
    auto wasHeld = false;

    for (auto index = 0; index < ticks; index++)
    {
        auto now = scenario;
        if (loadAfterTick >= 0 && index >= loadAfterTick)
        {
            now.load = loadAfter;
        }

        const auto next = advance(shaft, now, scheme, dt, x, v);
        result.ticks.push_back(next);
        result.impulse += next.impulse;
        result.work += next.impulse * 0.5 * (v + next.v);
        result.boundaryError = std::max({result.boundaryError, next.x - shaft.compression, -shaft.extension - next.x});

        const auto held = next.limit != RangeLimit::None;
        if (held)
        {
            result.heldTicks++;
            if (wasHeld)
            {
                result.heldVelocity = std::max(result.heldVelocity, std::abs(next.v));
            }
            result.reactionMin = std::min(result.reactionMin, next.reaction);
            result.reactionMax = std::max(result.reactionMax, next.reaction);
            if (!wasHeld)
            {
                result.contacts++;
            }
            if (result.firstHeld < 0)
            {
                result.firstHeld = index;
            }
        }
        else if (result.firstHeld >= 0 && result.firstFree < 0)
        {
            result.firstFree = index;
        }

        wasHeld = held;
        x = next.x;
        v = next.v;
    }

    result.kineticOut = 0.5 * shaft.mass * v * v;
    return result;
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

// The Golf with no air on it: the one external force the chassis then feels in the air is gravity,
// which is what a momentum ledger wants to be closed against.
[[nodiscard]] VehicleSetup golfInVacuum()
{
    auto setup = golfGtiMk7().value();
    setup.aero.clear();
    return setup;
}

// What one tick of the car did to the momentum ledger. The corners' momentum is the unsprung
// masses' along the wheel centres' own Jacobians, in the world, at the orientation the tick read;
// the chassis's is its mass times its velocity; gravity on the whole car is the external impulse.
struct Ledger
{
    glm::dvec3 residual{0.0}; // Δ(chassis) + Σ Δ(corners) − gravity·dt, N·s
    std::array<double, cornerCount> accelerationError{}; // |Q/I · dt − Δq̇|, rad/s
};

// The car's true momentum from a state (2026-09-08 latest of all (d), docs/chassis-kinematic-reaction-brief.md):
// the chassis's, plus every unsprung mass's beyond its design point, `m·[ω × (r − r_design) + R·C'·q̇]`,
// from a fresh solve at the corner's angle. Before (d) the ledger carried the `R·C'·q̇` part alone.
[[nodiscard]] glm::dvec3 trueMomentum(const VehicleSetup& setup, const VehicleState& state)
{
    const auto omega = raceengine::angularVelocity(state.chassis);
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
                                const double dt, raceengine::VehicleStep& stepped)
{
    const auto before = state;
    const auto momentumBefore = trueMomentum(setup, state);
    const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, dt);
    REQUIRE(result.has_value());
    stepped = result.value();

    auto ledger = Ledger{};
    ledger.residual = trueMomentum(setup, state) - momentumBefore - glm::dvec3(0.0, -gravity * state.chassis.mass * dt, 0.0);

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& solution = stepped.corners[index];
        const auto deltaRate = state.corners[index].wishboneRate - before.corners[index].wishboneRate;

        // The road's impulse, where there is a road: the load along the patch normal and the
        // in-plane forces along the patch frame, exactly as the force pass applies them.
        if (solution.patch.inContact)
        {
            ledger.residual -= (solution.forces.tireVertical * solution.patch.normal +
                                solution.contact.tyre.longitudinal * solution.contact.forward +
                                solution.contact.tyre.lateral * solution.contact.lateral) *
                               dt;
        }
        ledger.accelerationError[index] =
            std::abs(solution.generalisedForce / solution.generalisedInertia * dt - deltaRate);
    }

    return ledger;
}

} // namespace

// =================================================================================================
// Element proofs
// =================================================================================================

TEST_CASE("the range is inert inside it: the step's rate comes back bit for bit",
          "[physics][suspension][range-constraint]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    for (const auto& corner : golf->corners)
    {
        const auto& hardpoints = corner.hardpoints;
        for (const auto q : {0.0, 0.5 * hardpoints.droopAngle, 0.5 * hardpoints.bumpAngle, hardpoints.droopAngle,
                             hardpoints.bumpAngle})
        {
            for (const auto rate : {0.0, 1e-9, -1e-9, 0.37, -0.37, 3.0, -3.0})
            {
                const auto solved = constrainCornerRange(RangeConstraintStep{.previousAngle = q,
                                                                             .rate = rate,
                                                                             .droopAngle = hardpoints.droopAngle,
                                                                             .bumpAngle = hardpoints.bumpAngle,
                                                                             .deltaTime = tick});
                const auto predicted = q + rate * tick;
                if (predicted >= hardpoints.droopAngle && predicted <= hardpoints.bumpAngle)
                {
                    // Inside, or landing exactly on a limit: the rate is the step's own, the same bits.
                    REQUIRE(solved.limit == RangeLimit::None);
                    REQUIRE(solved.rate == rate);
                }
                else
                {
                    // Outside: the limit that would have been crossed, and a rate that lands on it,
                    // never pulling — the same sign as the free rate and smaller in magnitude.
                    REQUIRE(solved.limit ==
                            (predicted > hardpoints.bumpAngle ? RangeLimit::Bump : RangeLimit::Droop));
                    REQUIRE(solved.rate * rate >= 0.0);
                    REQUIRE(std::abs(solved.rate) <= std::abs(rate));
                    const auto limit = solved.limit == RangeLimit::Bump ? hardpoints.bumpAngle : hardpoints.droopAngle;
                    REQUIRE(q + solved.rate * tick == Catch::Approx(limit).margin(1e-15));
                }
            }
        }
    }
}

TEST_CASE("a free shaft meets its droop limit and its bump limit: the impulse is its momentum, the position is the "
          "limit, the work is never positive",
          "[physics][suspension][range-constraint]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    const auto shaft = shaftOf(golf->corners[2]);

    std::printf("\n=== free impact, Golf rear shaft: mass %.3f kg, limits −%.2f / +%.2f mm ===\n", shaft.mass,
                shaft.extension * 1000.0, shaft.compression * 1000.0);
    std::printf("  %-6s %8s %8s %10s %10s %10s %10s %8s %8s %8s\n", "limit", "v0 m/s", "held@", "impulse", "m·(0−v0)",
                "boundary", "work J", "KE in", "KE out", "v held");

    for (const auto v0 : {-0.5, 0.5, -2.0, 2.0})
    {
        const auto result = run(shaft, Scenario{}, Scheme::Constraint, tick, 720, 0.0, v0);
        const auto limit = v0 < 0.0 ? "droop" : "bump";

        std::printf("  %-6s %8.3f %8d %10.5f %10.5f %10.2e %10.5f %8.4f %8.4f %8.2e\n", limit, v0, result.firstHeld,
                    result.impulse, -shaft.mass * v0, result.boundaryError, result.work, result.kineticIn,
                    result.kineticOut, result.heldVelocity);

        // The shaft arrives and stops; the whole of its momentum is the constraint's impulse, signed
        // away from the limit; nothing goes past the limit by a single bit; the constraint removed
        // exactly the shaft's kinetic energy and never added any; held, the shaft does not move.
        REQUIRE(result.firstHeld > 0);
        REQUIRE(result.impulse == Catch::Approx(-shaft.mass * v0).epsilon(1e-12));
        REQUIRE(result.boundaryError == 0.0);
        REQUIRE(result.work <= 0.0);
        REQUIRE(result.work == Catch::Approx(-result.kineticIn).epsilon(1e-9));
        REQUIRE(result.kineticOut == 0.0);
        REQUIRE(result.heldVelocity == 0.0);
        REQUIRE(result.contacts == 1);

        // And with nothing pushing it, the shaft rests ON the limit and is not held by it: the free
        // step from rest lands exactly on the limit, which is inside, so λ is zero — complementarity.
        REQUIRE(result.ticks.back().limit == RangeLimit::None);
        REQUIRE(result.ticks.back().v == 0.0);
        REQUIRE(result.ticks.back().x == (v0 < 0.0 ? -shaft.extension : shaft.compression));

        // The reaction points away from the limit and only away from it.
        for (const auto& t : result.ticks)
        {
            REQUIRE((v0 < 0.0 ? t.reaction >= 0.0 : t.reaction <= 0.0));
        }
    }
}

TEST_CASE("a constant load holds the shaft on its limit without chatter, and a reversed one releases it on the first tick",
          "[physics][suspension][range-constraint]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    const auto shaft = shaftOf(golf->corners[2]);

    std::printf("\n=== static hold and release, Golf rear shaft ===\n");
    std::printf("  %-6s %8s %8s %8s %10s %10s %10s %8s %8s\n", "limit", "load N", "held", "entries", "F min", "F max",
                "v held", "release", "KE out");

    for (const auto load : {1278.0, -1278.0, 300.0, -300.0})
    {
        // The load on for four seconds — long enough for any limit cycle to show — then reversed,
        // and twenty ticks watched after that (a reversed load on a free shaft would reach the
        // other limit in a tenth of a second, which is a different case).
        const auto result = run(shaft, Scenario{.load = load}, Scheme::Constraint, tick, 1460, 0.0, 0.0, -load, 1440);
        const auto limit = load > 0.0 ? "droop" : "bump";

        std::printf("  %-6s %8.0f %8d %8d %10.3f %10.3f %10.2e %8d %8.2e\n", limit, load, result.heldTicks, result.contacts,
                    result.reactionMin, result.reactionMax, result.heldVelocity, result.firstFree - 1439,
                    result.kineticOut);

        // One entry, held every tick from arrival to reversal with the shaft exactly still and the
        // reaction exactly the load (the arrival tick's impulse aside), released on the very tick the
        // load reverses — with no adhesion: the reaction is never signed toward the limit.
        REQUIRE(result.contacts == 1);
        REQUIRE(result.firstHeld > 0);
        REQUIRE(result.heldTicks == 1440 - result.firstHeld);
        REQUIRE(result.heldVelocity == 0.0);
        REQUIRE(result.firstFree == 1440);
        // From two ticks after the arrival (the arrival lands at its own rate; the tick after kills
        // that rate; from then on the shaft is still): the reaction is the load, exactly.
        for (auto index = result.firstHeld + 2; index < 1440; index++)
        {
            const auto& t = result.ticks[static_cast<std::size_t>(index)];
            REQUIRE(t.v == 0.0);
            REQUIRE(t.reaction == Catch::Approx(load).epsilon(1e-12));
            REQUIRE(t.q == 0.0);
        }
        for (const auto& t : result.ticks)
        {
            REQUIRE((load > 0.0 ? t.reaction >= 0.0 : t.reaction <= 0.0));
        }
        // After the reversal the shaft is moving inward and nothing holds it.
        REQUIRE(result.ticks.back().limit == RangeLimit::None);
        REQUIRE((load > 0.0 ? result.ticks.back().v > 0.0 : result.ticks.back().v < 0.0));
    }
}

TEST_CASE("the soft stop stays a separate element: it carries what it can, and the limit carries only the residual",
          "[physics][suspension][range-constraint]")
{
    // The Golf's rear droop stop (gap 40 mm, reach 653 N at the 52.03 mm limit) in front of the
    // limit. A load inside the stop's reach settles on the stop and never reaches the limit; one
    // beyond it runs through the stop to the limit, where the balance is the stop's elastic law at
    // full compression plus the constraint, and nothing viscous.
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    const auto shaft = shaftOf(golf->corners[2]);
    const auto& stop = shaft.corner.droopStop;
    const auto reach = stop.elasticForce(shaft.extension - stop.gap);

    std::printf("\n=== soft stop before the hard limit, Golf rear: stop reach at the limit %.1f N ===\n", reach);
    std::printf("  %-8s %8s %8s %10s %10s %10s %10s\n", "load N", "held", "x end mm", "stop N", "viscous", "F limit", "sum");

    // The stop stores under two joules over its whole 12 mm, so a shaft arriving from design under
    // any load worth holding blows through it: the inside case is applied at the touch point, from
    // rest, and the through case from design.
    for (const auto load : {100.0, 1278.0})
    {
        const auto result = run(shaft, Scenario{.stops = true, .load = load}, Scheme::Constraint, tick, 2160,
                                load < reach ? -stop.gap : 0.0, 0.0);
        const auto& last = result.ticks.back();
        const auto elastic = -stop.elasticForce(-last.x - stop.gap);
        const auto viscous = last.droop - elastic;

        std::printf("  %8.0f %8d %10.3f %10.3f %10.2e %10.3f %10.4f\n", load, result.heldTicks, last.x * 1000.0, last.droop,
                    viscous, last.reaction, load + last.droop - last.reaction);

        if (load < reach)
        {
            REQUIRE(result.heldTicks == 0);
            REQUIRE(last.limit == RangeLimit::None);
            REQUIRE(-last.x < shaft.extension);
            REQUIRE(-last.x > stop.gap);
            REQUIRE(last.droop == Catch::Approx(-load).margin(0.5)); // the stop alone carries it (settling)
        }
        else
        {
            REQUIRE(last.limit == RangeLimit::Droop);
            REQUIRE(last.v == 0.0);
            REQUIRE(-last.x == shaft.extension);
            REQUIRE(last.droop == Catch::Approx(-reach).epsilon(1e-12)); // the elastic law at the limit, no viscous share
            REQUIRE(viscous == 0.0);
            REQUIRE(last.reaction == Catch::Approx(load - reach).epsilon(1e-12));
        }
    }
}

TEST_CASE("held on its limit the shaft's damper reads its force at rest, where the clamp before read the force of a velocity it had thrown away",
          "[physics][suspension][range-constraint]")
{
    // The audit's arm B (the rear droop stop disabled) under the 0.83 g stop: 34 ticks on the
    // limit with the shaft stationary and a damper force of up to 586 N on it. Here the same
    // shaft — the Golf's own rear damper curve and seal friction — is driven onto its droop limit
    // by a constant load and held; on every held tick after the arrival the shipped block reads the
    // damper's force at zero velocity, and the block of the build before reads the force of the free
    // rate the clamp then discarded.
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    const auto shaft = shaftOf(golf->corners[2]);
    const auto atRest = damperAt(shaft, 0.0).force;

    std::printf("\n=== damper on a held shaft, Golf rear (damper force at rest %.3f N) ===\n", atRest);
    std::printf("  %-10s %8s %8s %12s %12s %12s %10s\n", "scheme", "load N", "held", "|damper| max", "damper mean",
                "|v| held", "q max");

    for (const auto load : {625.0, 1278.0})
    {
        for (const auto scheme : {Scheme::Clamp, Scheme::Constraint})
        {
            const auto result = run(shaft, Scenario{.damper = true, .load = load}, scheme, tick, 1080, 0.0, 0.0);

            auto worst = 0.0;
            auto mean = 0.0;
            auto velocity = 0.0;
            auto q = 0.0;
            auto counted = 0;
            for (auto index = result.firstHeld + 2; index < 1080; index++)
            {
                const auto& t = result.ticks[static_cast<std::size_t>(index)];
                worst = std::max(worst, std::abs(t.damper));
                mean += t.damper;
                velocity = std::max(velocity, std::abs(t.v));
                q = std::max(q, std::abs(t.q));
                counted++;
            }
            mean /= static_cast<double>(std::max(counted, 1));

            std::printf("  %-10s %8.0f %8d %12.3f %12.3f %12.2e %10.3f\n", scheme == Scheme::Clamp ? "clamp" : "constraint",
                        load, counted, worst, mean, velocity, q);

            REQUIRE(counted > 600);
            REQUIRE(velocity == 0.0);
            if (scheme == Scheme::Constraint)
            {
                // The force at rest, exactly, and a corner whose acceleration is exactly zero.
                REQUIRE(worst == std::abs(atRest));
                REQUIRE(q == 0.0);
            }
            else
            {
                // The artefact, reproduced: a stationary shaft reporting the force of a moving one,
                // and a chassis handed the acceleration of a wheel that did not accelerate.
                REQUIRE(worst > 100.0);
                REQUIRE(q > 100.0);
            }
        }
    }
}

TEST_CASE("the constraint is stable across the timestep: no overshoot, the same impulse, energy only ever removed",
          "[physics][suspension][range-constraint]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    const auto shaft = shaftOf(golf->corners[2]);

    // The whole corner on its shaft — spring with its preload, the production damper, both stops —
    // arriving at −0.5 m/s under a 625 N load into the droop limit, held for 1.5 s, then the load
    // reversed. Across four rates: nothing ever past the limit, the constraint's work never positive,
    // one contact, the shaft exactly still while held, release on the reversal tick, and the impulse
    // over the whole event agreeing across the rates — it is the physical quantity (the momentum the
    // shaft arrived with plus what the net force fed in while it was held), and a scheme that
    // depended on the rate would not agree with itself.
    std::printf("\n=== timestep sweep, Golf rear shaft: −0.5 m/s into the droop limit under a 625 N load, then reversed ===\n");
    std::printf("  %8s %8s %10s %10s %10s %10s %8s %8s\n", "Hz", "held@ s", "impulse", "F held", "boundary", "work J",
                "entries", "release");

    auto impulses = std::vector<double>{};
    for (const auto hz : {120.0, 360.0, 720.0, 2880.0})
    {
        const auto dt = 1.0 / hz;
        const auto ticks = static_cast<int>(std::lround(2.0 * hz));
        const auto reverse = static_cast<int>(std::lround(1.5 * hz));
        const auto load = 625.0;
        const auto result =
            run(shaft, Scenario{.spring = true, .damper = true, .stops = true, .load = load}, Scheme::Constraint, dt,
                ticks, 0.0, -0.5, -load, reverse);
        const auto heldForce = result.ticks[static_cast<std::size_t>(reverse - 1)].reaction;

        std::printf("  %8.0f %8.4f %10.4f %10.3f %10.2e %10.5f %8d %8d\n", hz,
                    static_cast<double>(result.firstHeld) * dt, result.impulse, heldForce, result.boundaryError,
                    result.work, result.contacts, result.firstFree - reverse);

        REQUIRE(result.boundaryError == 0.0);
        REQUIRE(result.work <= 0.0);
        REQUIRE(result.contacts == 1);
        REQUIRE(result.heldVelocity == 0.0);
        REQUIRE(result.firstFree == reverse);
        REQUIRE(result.firstHeld > 0);
        REQUIRE(static_cast<double>(result.firstHeld) * dt < 0.5);
        impulses.push_back(result.impulse);
    }

    for (const auto impulse : impulses)
    {
        REQUIRE(impulse == Catch::Approx(impulses.back()).epsilon(0.02));
    }
}

// =================================================================================================
// Whole-car proofs
// =================================================================================================

TEST_CASE("in the air the car falls at g with its wheels hanging from their limits, and the momentum ledger closes on every tick",
          "[physics][suspension][range-constraint]")
{
    // The Golf, air removed, dropped from 30 m: the springs push every wheel out through its droop
    // stop to its droop limit within the first tenth of a second, and the car falls for two seconds
    // with all four hanging. The chassis is then a free body under gravity alone, and the ledger —
    // the chassis's momentum plus the four unsprung masses' along their own Jacobians, against the
    // gravity impulse — has to close to rounding on every tick, the arrival ticks included. On the
    // build before it did not: the clamp threw the wheel's momentum away and the chassis kept the
    // reaction of an acceleration that never happened, so the car fell at less than g.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());
    const auto setup = golfInVacuum();

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 30.0, 200.0);

    auto worstResidual = 0.0;
    auto worstAcceleration = 0.0;
    auto heldTicks = 0;
    auto arrivals = 0;
    auto held = std::array<bool, cornerCount>{};
    auto reactionAtWheel = std::array<double, cornerCount>{};
    auto fallRateError = 0.0;

    std::printf("\n=== the Golf in the air, 2 s ===\n");
    for (auto step = 0; step < 720; step++)
    {
        const auto before = state.chassis.linearVelocity.y;
        auto stepped = raceengine::VehicleStep{};
        const auto ledger = stepLedger(setup, state, world.value(), tick, stepped);

        worstResidual = std::max(worstResidual, glm::length(ledger.residual));
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            worstAcceleration = std::max(worstAcceleration, ledger.accelerationError[index]);
            const auto& solution = stepped.corners[index];
            const auto now = solution.rangeLimit != RangeLimit::None;
            heldTicks += now ? 1 : 0;
            arrivals += now && !held[index] ? 1 : 0;
            held[index] = now;
            reactionAtWheel[index] = solution.forces.rangeLimit;
            REQUIRE(solution.forces.tireVertical == 0.0);
            if (now)
            {
                REQUIRE(solution.rangeLimit == RangeLimit::Droop);
                REQUIRE(solution.forces.rangeLimit > 0.0); // pulling the wheel up: it hangs
            }
        }

        // Once all four hang the chassis is in free fall, exactly. The rears arrive at a third of a
        // second rather than a fifth since the corner reads the chassis's acceleration (2026-09-08
        // latest of all, docs/frame-acceleration-brief.md): a wheel in free fall is pushed out by its
        // spring's preload alone, not by its weight as well.
        if (step > 360)
        {
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                REQUIRE(held[index]);
                REQUIRE(state.corners[index].wishboneRate == 0.0);
                REQUIRE(state.corners[index].wishboneAngle == setup.corners[index].hardpoints.droopAngle);
            }
            fallRateError = std::max(fallRateError, std::abs((state.chassis.linearVelocity.y - before) + gravity * tick));
        }

        if (step == 36 || step == 72 || step == 719)
        {
            std::printf("  tick %3d: vy %.4f m/s (free fall %.4f), residual %.2e N·s, held corners %d, reaction at the wheel",
                        step + 1, state.chassis.linearVelocity.y, -gravity * tick * static_cast<double>(step + 1),
                        glm::length(ledger.residual),
                        static_cast<int>(std::count(held.begin(), held.end(), true)));
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                std::printf(" %s %.1f", cornerAbbreviation(static_cast<Corner>(index)), reactionAtWheel[index]);
            }
            std::printf(" N\n");
        }
    }

    std::printf("  worst ledger residual %.3e N·s, worst |Q/I·dt − Δq̇| %.3e rad/s, held corner-ticks %d, arrivals %d, "
                "free-fall error %.3e m/s per tick\n",
                worstResidual, worstAcceleration, heldTicks, arrivals, fallRateError);

    // To the second order of the displaced wheels' explicit inertial force on the true ledger
    // (docs/chassis-kinematic-reaction-brief.md); rounding where nothing turns.
    REQUIRE(worstResidual < 1e-3);
    REQUIRE(worstAcceleration < 1e-9);
    REQUIRE(arrivals == 4);
    REQUIRE(heldTicks > 1440);
    REQUIRE(fallRateError < 1e-12);

    // What each hanging wheel pulls the body down with: the spring's push at full droop less the
    // droop stop's reach — the audit's 1278 − 653 ≈ 625 N at the rear, less the 422 N a wheel in
    // free fall does not weigh (2026-09-08 latest of all, docs/frame-acceleration-brief.md: the
    // corner reads the chassis's acceleration, and the build before read 629 here), within the
    // Jacobians' few per cent. With `frameAcceleration` off the 625 comes back.
    REQUIRE(reactionAtWheel[2] == Catch::Approx(207.0).margin(40.0));
    REQUIRE(reactionAtWheel[3] == Catch::Approx(207.0).margin(40.0));
}

TEST_CASE("held against its droop limit by the road, a corner's tyre loads sum to the car's weight and the corner does not chatter",
          "[physics][suspension][range-constraint]")
{
    // The audit's whole-car quasi-static unload, one row: the rear droop stop disabled (arm B, so the
    // limit alone carries the hanging wheel) and 130 kg at 8 m behind the front axle, settled. On the
    // build before, this row read the four tyre loads at the weight minus 1601 N and a pitch of
    // −11.8°; the corner was on the limit 360 ticks of 360 with a damper force on a stationary shaft.
    const auto guard = JoltGuard{};
    const auto world = plate(1600.0);
    REQUIRE(world.has_value());

    auto setup = golfInVacuum();
    for (const auto index : {std::size_t{2}, std::size_t{3}})
    {
        setup.corners[index].droopStop.rate = 0.0;
        setup.corners[index].droopStop.damping = 0.0;
    }
    setup.sprung.push_back(MassComponent{.mass = 130.0,
                                         .centre = glm::dvec3(0.0, 0.50, frontAxle + 8.0),
                                         .inertia = glm::dmat3(0.0)});

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 800.0);

    auto stepped = raceengine::VehicleStep{};
    for (auto step = 0; step < 2520; step++)
    {
        const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(result.has_value());
        stepped = result.value();
    }

    // The last second: held every tick, still, the chassis reaction reading the corner's zero
    // acceleration, the momentum ledger — road, gravity, chassis, four unsprung masses — closing on
    // every tick, and the audit's own statistic, the four tyre loads against the car's weight, at
    // the weight on average (per tick it also carries the body's residual ring-down, which is
    // physics and not the reaction).
    auto heldTicks = 0;
    auto entries = 0;
    auto sumError = 0.0;
    auto worstResidual = 0.0;
    auto worstVelocity = 0.0;
    auto worstDamper = 0.0;
    auto worstQ = 0.0;
    auto reaction = 0.0;
    auto wasHeld = true;
    for (auto step = 0; step < 360; step++)
    {
        const auto ledger = stepLedger(setup, state, world.value(), tick, stepped);
        worstResidual = std::max(worstResidual, glm::length(ledger.residual));

        auto sum = 0.0;
        for (const auto& solution : stepped.corners)
        {
            sum += solution.forces.tireVertical;
        }
        sumError += (sum - state.chassis.mass * gravity) / 360.0;

        const auto& rear = stepped.corners[2];
        const auto held = rear.rangeLimit == RangeLimit::Droop;
        heldTicks += held ? 1 : 0;
        entries += held && !wasHeld ? 1 : 0;
        wasHeld = held;
        worstVelocity = std::max(worstVelocity, std::abs(state.corners[2].wishboneRate));
        worstDamper = std::max(worstDamper, std::abs(rear.forces.damper));
        worstQ = std::max(worstQ, std::abs(rear.generalisedForce));
        reaction = rear.forces.rangeLimit;
    }

    std::printf("\n=== 130 kg behind the rear axle, droop stop disabled, settled 8 s: rear held %d/360 (entries %d), "
                "ΣFz − W mean %.2f N (the build before: −1601), ledger residual max %.2e N·s, |q̇| max %.2e, |damper| max %.2e N, "
                "|Q| max %.2e N·m, reaction at the wheel %.1f N, pitch %.3f° (the build before: −11.8), rear Fz %.1f N ===\n",
                heldTicks, entries, sumError, worstResidual, worstVelocity, worstDamper, worstQ, reaction,
                stepped.telemetry.pitch * degrees, stepped.corners[2].forces.tireVertical);

    REQUIRE(heldTicks == 360);
    REQUIRE(entries == 0);
    REQUIRE(std::abs(sumError) < 25.0);
    // To the second order of the displaced wheels' explicit inertial force on the true ledger
    // (docs/chassis-kinematic-reaction-brief.md); rounding where nothing turns.
    REQUIRE(worstResidual < 1e-3);
    REQUIRE(worstVelocity == 0.0);
    REQUIRE(worstDamper == 0.0);
    REQUIRE(worstQ < 1e-9);
    REQUIRE(reaction > 100.0);
    REQUIRE(std::abs(stepped.telemetry.pitch * degrees) < 3.0);
}

TEST_CASE("held against its bump limit by a load, a corner is pushed back with the same accounting, and releases the tick the load goes",
          "[physics][suspension][range-constraint]")
{
    // The other sign: the rear bump stop disabled (so the spring alone stands between the load and
    // the limit) and 700 kg over the rear axle, which puts both rears on their bump limits. Held
    // still, the reaction negative (toward droop), the loads summing to the weight; then the mass
    // removed, and the corners leave their limits on the first tick.
    const auto guard = JoltGuard{};
    const auto world = plate(1600.0);
    REQUIRE(world.has_value());

    auto loaded = golfInVacuum();
    for (const auto index : {std::size_t{2}, std::size_t{3}})
    {
        loaded.corners[index].bumpStop.rate = 0.0;
        loaded.corners[index].bumpStop.damping = 0.0;
    }
    auto unloaded = loaded;
    loaded.sprung.push_back(MassComponent{.mass = 700.0,
                                          .centre = glm::dvec3(0.0, 0.50, -frontAxle),
                                          .inertia = glm::dmat3(0.0)});

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 800.0);

    for (auto step = 0; step < 2520; step++)
    {
        REQUIRE(stepVehicle(loaded, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    auto heldTicks = 0;
    auto sumError = 0.0;
    auto worstResidual = 0.0;
    auto worstVelocity = 0.0;
    auto reaction = 0.0;
    auto stepped = raceengine::VehicleStep{};
    for (auto step = 0; step < 360; step++)
    {
        const auto ledger = stepLedger(loaded, state, world.value(), tick, stepped);
        worstResidual = std::max(worstResidual, glm::length(ledger.residual));

        auto sum = 0.0;
        for (const auto& solution : stepped.corners)
        {
            sum += solution.forces.tireVertical;
        }
        sumError += (sum - state.chassis.mass * gravity) / 360.0;
        heldTicks += stepped.corners[2].rangeLimit == RangeLimit::Bump && stepped.corners[3].rangeLimit == RangeLimit::Bump ? 1 : 0;
        worstVelocity = std::max({worstVelocity, std::abs(state.corners[2].wishboneRate), std::abs(state.corners[3].wishboneRate)});
        reaction = stepped.corners[2].forces.rangeLimit;
    }

    std::printf("\n=== 700 kg over the rear axle, bump stop disabled, settled 8 s: both rears held %d/360, ΣFz − W mean %.2f N, "
                "ledger residual max %.2e N·s, |q̇| max %.2e, reaction at the wheel %.1f N, rear Fz %.1f N, rear travel %.2f mm ===\n",
                heldTicks, sumError, worstResidual, worstVelocity, reaction, stepped.corners[2].forces.tireVertical,
                stepped.corners[2].suspension.wheelTravel * 1000.0);

    REQUIRE(heldTicks == 360);
    REQUIRE(std::abs(sumError) < 25.0);
    // To the second order of the displaced wheels' explicit inertial force on the true ledger
    // (docs/chassis-kinematic-reaction-brief.md); rounding where nothing turns.
    REQUIRE(worstResidual < 1e-3);
    REQUIRE(worstVelocity == 0.0);
    REQUIRE(reaction < -100.0);

    // The mass gone. The road does not let go on one tick — the tyre is still carrying the load
    // until the body has risen — so the corner stays held while the reaction falls, leaves the limit
    // once the free step points inward, and once free is never held again: no adhesion, no
    // re-entry.
    auto releaseTick = -1;
    auto reHeld = 0;
    auto lastReaction = reaction;
    auto reactionRose = 0;
    for (auto step = 0; step < 360; step++)
    {
        const auto released = stepVehicle(unloaded, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(released.has_value());
        const auto held = released->corners[2].rangeLimit == RangeLimit::Bump;
        if (held && releaseTick < 0)
        {
            reactionRose += released->corners[2].forces.rangeLimit < lastReaction - 1e-9 ? 1 : 0;
            lastReaction = released->corners[2].forces.rangeLimit;
        }
        if (!held && releaseTick < 0)
        {
            releaseTick = step;
        }
        reHeld += held && releaseTick >= 0 ? 1 : 0;
    }

    std::printf("  mass removed: the rear left its bump limit after %d ticks (the reaction fell monotonically: %s), "
                "re-held afterwards %d ticks, rear rate then %.4f rad/s\n",
                releaseTick, reactionRose == 0 ? "yes" : "no", reHeld, state.corners[2].wishboneRate);

    REQUIRE(releaseTick >= 0);
    REQUIRE(releaseTick < 180);
    REQUIRE(reHeld == 0);
}

TEST_CASE("the fixtures' settle from 0.52 m never reaches a limit, and the 1.02 m drop does",
          "[physics][suspension][range-constraint]")
{
    // What makes the whole characterisation byte-inert under this change: every scenario's settle
    // starts the car at 0.52 m, three centimetres below its rest height, and the bounce that follows
    // reaches neither a droop limit nor — measured here, against the note that said every settle
    // engages all four droop stops — a droop stop. The integrator test's 1.02 m drop reaches the
    // limits, on all four corners.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());
    const auto setup = golfGtiMk7().value();

    std::printf("\n=== settle drops ===\n");
    for (const auto height : {0.52, 1.02})
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, height, 200.0);

        auto limitTicks = std::array<int, cornerCount>{};
        auto stopTicks = std::array<int, cornerCount>{};
        auto closest = std::array<double, cornerCount>{1e9, 1e9, 1e9, 1e9};
        for (auto step = 0; step < 2160; step++)
        {
            const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(result.has_value());
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                limitTicks[index] += result->corners[index].rangeLimit != RangeLimit::None ? 1 : 0;
                stopTicks[index] += result->corners[index].forces.droopStop != 0.0 ? 1 : 0;
                closest[index] = std::min(closest[index], state.corners[index].wishboneAngle -
                                                              setup.corners[index].hardpoints.droopAngle);
            }
        }

        std::printf("  from %.2f m:", height);
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            std::printf("  %s limit %d ticks, droop stop %d, closest %.2f mrad", cornerAbbreviation(static_cast<Corner>(index)),
                        limitTicks[index], stopTicks[index], closest[index] * 1000.0);
        }
        std::printf("\n");

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            if (height < 1.0)
            {
                REQUIRE(limitTicks[index] == 0);
                REQUIRE(closest[index] > 0.0);
            }
            else
            {
                REQUIRE(stopTicks[index] > 0);
                REQUIRE(limitTicks[index] > 0);
            }
        }
    }
}

// =================================================================================================
// Forensic probes, hidden: the settle's trend against a limit, and the fleet
// =================================================================================================

TEST_CASE("the audit's unload rows, tick by tick: does a corner against its limit converge or cycle", "[.range-constraint-forensic]")
{
    // The droop audit's whole-car quasi-static unload at the masses where the rear reached its
    // limit, on the production stop (arm A) and with it disabled (arm B), settled ten seconds with
    // every second's limit entries, held ticks, ΣFz − W and the rear's excursion printed — the
    // difference between a ring-down that touches the limit on its way to rest and a limit cycle.
    const auto guard = JoltGuard{};
    const auto world = plate(1600.0);
    REQUIRE(world.has_value());

    for (const auto disabled : {false, true})
    {
        for (const auto kilograms : {130.0, 140.0, 150.0, 160.0})
        {
            auto setup = golfGtiMk7().value();
            if (disabled)
            {
                for (const auto index : {std::size_t{2}, std::size_t{3}})
                {
                    setup.corners[index].droopStop.rate = 0.0;
                    setup.corners[index].droopStop.damping = 0.0;
                }
            }
            setup.sprung.push_back(MassComponent{.mass = kilograms,
                                                 .centre = glm::dvec3(0.0, 0.50, frontAxle + 8.0),
                                                 .inertia = glm::dmat3(0.0)});

            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, 0.52, 800.0);

            std::printf("\n=== arm %s, %.0f kg at 8 m behind the front axle ===\n", disabled ? "B (stop disabled)" : "A (production stop)",
                        kilograms);
            std::printf("  %4s %6s %8s %10s %10s %10s %9s %9s %8s\n", "s", "held", "entries", "sumFz-W", "F lim max", "q min",
                        "q max mrad", "pitch", "Fz RL");

            for (auto second = 0; second < 10; second++)
            {
                auto heldTicks = 0;
                auto entries = 0;
                auto sumError = 0.0;
                auto reaction = 0.0;
                auto qMin = 1e9;
                auto qMax = -1e9;
                auto wasHeld = false;
                auto pitch = 0.0;
                auto fz = 0.0;
                for (auto step = 0; step < 360; step++)
                {
                    const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
                    REQUIRE(result.has_value());
                    auto sum = 0.0;
                    for (const auto& solution : result->corners)
                    {
                        sum += solution.forces.tireVertical;
                    }
                    sumError += sum - state.chassis.mass * gravity;
                    const auto held = result->corners[2].rangeLimit == RangeLimit::Droop;
                    heldTicks += held ? 1 : 0;
                    entries += held && !wasHeld ? 1 : 0;
                    wasHeld = held;
                    reaction = std::max(reaction, result->corners[2].forces.rangeLimit);
                    const auto q = state.corners[2].wishboneAngle - setup.corners[2].hardpoints.droopAngle;
                    qMin = std::min(qMin, q);
                    qMax = std::max(qMax, q);
                    pitch = result->telemetry.pitch * degrees;
                    fz = result->corners[2].forces.tireVertical;
                }
                std::printf("  %4d %6d %8d %10.2f %10.1f %10.4f %9.4f %9.3f %8.1f\n", second + 1, heldTicks, entries,
                            sumError / 360.0, reaction, qMin * 1000.0, qMax * 1000.0, pitch, fz);
            }
        }
    }
}

TEST_CASE("fleet: which corners of each published car reach a range limit, and what the limit then carries", "[.range-constraint-fleet]")
{
    // The two published setups — the Golf (strut front, multilink rear) and the placeholder sedan
    // (short-long arm both ends) — through the fixtures' 0.52 m settle, the 1.02 m drop, and two
    // seconds in the air; per corner, the ticks on each limit and the reaction at the wheel. Reported,
    // not tuned.
    const auto guard = JoltGuard{};
    const auto world = plate(400.0);
    REQUIRE(world.has_value());

    struct Car
    {
        const char* name;
        VehicleSetup setup;
    };
    const auto cars = std::array{Car{"Golf GTI Mk7", golfGtiMk7().value()}, Car{"placeholder sedan", placeholderSedan().value()}};

    for (const auto& car : cars)
    {
        std::printf("\n=== %s ===\n", car.name);
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto shaft = shaftOf(car.setup.corners[index]);
            const auto& corner = car.setup.corners[index];
            std::printf("  %s: range droop %.4f / bump %.4f rad = shaft −%.2f / +%.2f mm; droop stop gap %.1f mm reach %.1f N, "
                        "bump stop gap %.1f mm; spring at full droop %.1f N, unsprung weight %.1f N\n",
                        cornerAbbreviation(static_cast<Corner>(index)), corner.hardpoints.droopAngle,
                        corner.hardpoints.bumpAngle, shaft.extension * 1000.0, shaft.compression * 1000.0,
                        corner.droopStop.gap * 1000.0, corner.droopStop.elasticForce(shaft.extension - corner.droopStop.gap),
                        corner.bumpStop.gap * 1000.0,
                        raceengine::solveSpringForce(corner, solveCornerWithJacobian(corner.hardpoints, corner.hardpoints.droopAngle, 0.0).value()).force,
                        corner.unsprungMass * gravity);
        }

        for (const auto height : {0.52, 1.02, 30.0})
        {
            auto setup = car.setup;
            if (height > 10.0)
            {
                setup.aero.clear();
            }
            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, height, 200.0);

            auto droopTicks = std::array<int, cornerCount>{};
            auto bumpTicks = std::array<int, cornerCount>{};
            auto reaction = std::array<double, cornerCount>{};
            auto largest = std::array<double, cornerCount>{};
            const auto ticks = height > 10.0 ? 720 : 2160;
            for (auto step = 0; step < ticks; step++)
            {
                const auto result = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
                REQUIRE(result.has_value());
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    const auto& solution = result->corners[index];
                    droopTicks[index] += solution.rangeLimit == RangeLimit::Droop ? 1 : 0;
                    bumpTicks[index] += solution.rangeLimit == RangeLimit::Bump ? 1 : 0;
                    reaction[index] = solution.forces.rangeLimit;
                    largest[index] = std::abs(solution.forces.rangeLimit) > std::abs(largest[index]) ? solution.forces.rangeLimit
                                                                                                      : largest[index];
                }
            }

            std::printf("  %s %.2f m, %d ticks:", height > 10.0 ? "in the air from" : "dropped from", height, ticks);
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                std::printf("  %s droop %d bump %d ticks, reaction end %.1f largest %.1f N",
                            cornerAbbreviation(static_cast<Corner>(index)), droopTicks[index], bumpTicks[index],
                            reaction[index], largest[index]);
            }
            std::printf("; chassis y %.3f m\n", state.chassis.position.y);
        }
    }
}
