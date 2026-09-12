#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::CornerSetup;
using raceengine::damperElementOf;
using raceengine::golfGtiMk7;
using raceengine::solveCornerWithJacobian;
using raceengine::solveElement;
using raceengine::TravelStop;

// The bump/droop stop's dynamic term, at the element (docs/stop-element-brief.md, 2026-09-08).
//
// Two defects were measured on the shipped element by the suspension characterisation: **D1**, the
// placed viscous constant was stepped explicitly and a corner resting on its stop alternated
// 1729 N / 0 N on every tick, and **D2**, that constant put `damping · v` on the wheel in the tick
// contact became true — 0 → 28 kN at 0.7 m/s. The fix holds the authored elastic law and the gap
// exactly, scales the coefficient by the stop's own tangent stiffness so it enters from zero, and
// solves it in the corner's implicit divisor beside the damper's slope.
//
// These cases are the element-level statement of that, on a one-degree-of-freedom replica of the
// corner's velocity step written in shaft coordinates — the corner's generalised quantities are the
// shaft's times one Jacobian squared, so the replica and the corner run the same recurrence. The
// old law is kept here, in the replica only, as the control every table is read against. Since
// 2026-09-08 later the replica's `New` and `Zero` laws step the ordinary damper the corrected way
// (`solveCornerRate`, docs/damper-integrator-brief.md) and `Old` keeps the old damper step as well
// as the old stop law: it is the corner as it was, entire.
//
// Every table prints; `./EngineTests "[stop]" -s` shows them.

namespace
{

constexpr auto gravity = 9.80665;

// One corner reduced to its damper shaft: a mass, the corner's own damper, its spring and its bump
// stop, all referred to the shaft through the design-position Jacobians. `mass` is
// `m_u · |dC/dq|² / L'²`, which is the generalised inertia the force pass uses with the geometric
// load path on (the Golf's setting) brought onto the shaft.
struct Shaft
{
    const char* name = "";
    double mass = 0.0;
    double springRate = 0.0;
    double wheelPerShaft = 0.0; // |dWheelTravel / dShaftCompression| at design, for referring a wheel load
    raceengine::Curve damper;
    double friction = 0.0;
    double frictionSpeed = 0.0;
    TravelStop stop;
};

[[nodiscard]] Shaft shaftOf(const CornerSetup& corner, const char* name)
{
    const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
    REQUIRE(design.has_value());

    const auto element = damperElementOf(corner.hardpoints);
    const auto jacobian = solveElement(corner.hardpoints, element, 0.0).lengthPerAngle;
    const auto inertia = corner.unsprungMass * glm::dot(design->wheelCentrePerAngle, design->wheelCentrePerAngle);

    return Shaft{.name = name,
                 .mass = inertia / (jacobian * jacobian),
                 .springRate = corner.springRate,
                 .wheelPerShaft = std::abs(design->travelPerAngle / jacobian),
                 .damper = corner.damper,
                 .friction = corner.damperFriction,
                 .frictionSpeed = std::max(corner.damperFrictionSpeed, 1e-9),
                 .stop = corner.bumpStop};
}

// The corner's damper force and implicit slope, as `solveDamperForce` and `damperDampingCoefficient`
// form them on the shipped branch (friction stated, no velocity shape).
[[nodiscard]] double damperForce(const Shaft& shaft, const double velocity)
{
    const auto viscous = shaft.damper.at(velocity);

    return shaft.friction <= 0.0 ? viscous : viscous + shaft.friction * std::tanh(velocity / shaft.frictionSpeed);
}

[[nodiscard]] double damperSlope(const Shaft& shaft, const double velocity)
{
    const auto slope = (shaft.damper.at(velocity + 1e-4) - shaft.damper.at(velocity - 1e-4)) / 2e-4;
    if (shaft.friction <= 0.0)
    {
        return std::max(0.0, slope);
    }

    const auto shaped = std::tanh(velocity / shaft.frictionSpeed);

    return std::max(0.0, slope + (shaft.friction / shaft.frictionSpeed) * (1.0 - shaped * shaped));
}

// The law before 2026-09-08, written out: the elastic term, a constant viscous coefficient, and
// the push-only clamp over the sum. The control.
[[nodiscard]] double oldStopForce(const TravelStop& stop, const double past, const double velocity)
{
    if (past <= 0.0)
    {
        return 0.0;
    }

    return std::max(0.0, stop.elasticForce(past) + stop.damping * velocity);
}

enum class Law
{
    Old,
    New,
    Zero // the forensic control: the elastic law alone, no dynamic term at all
};

[[nodiscard]] const char* lawName(const Law law)
{
    return law == Law::Old ? "old" : law == Law::New ? "new" : "zero";
}

struct Tick
{
    double compression = 0.0; // of the shaft after the step, metres, positive in bump
    double velocity = 0.0;    // after the step, m/s, compression-positive
    double stopForce = 0.0;   // the force that acted, newtons, positive pushing out
    double viscous = 0.0;     // its viscous share
};

struct Scenario
{
    bool spring = false;
    bool damper = false;
    double load = 0.0; // a constant force into the stop, newtons on the shaft

    // The tyre as the corner's ground, referred onto the shaft: a spring about `equilibrium` and
    // the explicit vertical damping the force pass applies. Without it the replica's ground is
    // rigid and the corner has no wheel-hop loop — and it is that loop, with the tyre's damping
    // stepped explicitly beside the stop's, that the whole car's resting-on-stop alternation
    // lives in.
    double tyreRate = 0.0;
    double tyreDamping = 0.0;
    double equilibrium = 0.0;
};

// One tick of the corner's velocity step, in shaft coordinates. Old: the stop's whole force in the
// explicit numerator, the damper's slope in the divisor — and, since it is the corner before
// 2026-09-08 entire, the damper's whole force in the numerator too (the double application
// docs/damper-integrator-brief.md closed). New: the stop's elastic part in the numerator, its
// viscous coefficient in the divisor, the viscous share read off the solved velocity, the clamp as a
// re-solve without the stop — and the damper's affine intercept `F(v) − F'(v)·v` in the numerator
// beside its slope in the divisor, which is `solveCornerRate`'s recurrence on the shaft. Zero: the
// elastic part alone, on the same damper step.
[[nodiscard]] Tick advance(const Shaft& shaft, const Scenario& scenario, const Law law, const double dt,
                           const double compression, const double velocity)
{
    const auto past = compression - shaft.stop.gap;
    const auto other = scenario.load - (scenario.spring ? shaft.springRate * compression : 0.0) -
                       (scenario.damper ? damperForce(shaft, velocity) : 0.0) -
                       scenario.tyreRate * (compression - scenario.equilibrium) - scenario.tyreDamping * velocity;
    const auto slope = scenario.damper ? damperSlope(shaft, velocity) : 0.0;

    auto result = Tick{};

    if (law == Law::Old)
    {
        const auto stop = oldStopForce(shaft.stop, past, velocity);
        auto next = velocity + ((other - stop) / shaft.mass) * dt;
        next /= 1.0 + (slope / shaft.mass) * dt;
        result.velocity = next;
        result.stopForce = stop;
        result.viscous = past > 0.0 ? stop - shaft.stop.elasticForce(past) : 0.0;
    }
    else
    {
        const auto terms = shaft.stop.terms(past, velocity);
        const auto coefficient = law == Law::New ? terms.viscousCoefficient : 0.0;

        // The damper's intercept, on the shaft: `other` carries `−F(v)` whole, and the slope's own
        // share of it, `−F'(v)·v`, is put back because the divisor applies it at the new velocity.
        const auto intercept = slope * velocity;

        auto next = velocity + ((other + intercept - terms.explicitForce) / shaft.mass) * dt;
        next /= 1.0 + ((slope + coefficient) / shaft.mass) * dt;

        auto stop = terms.explicitForce + coefficient * next;
        auto viscous = coefficient * next;
        if (stop < 0.0)
        {
            next = velocity + ((other + intercept) / shaft.mass) * dt;
            next /= 1.0 + (slope / shaft.mass) * dt;
            stop = 0.0;
            viscous = 0.0;
        }

        result.velocity = next;
        result.stopForce = stop;
        result.viscous = viscous;
    }

    result.compression = compression + result.velocity * dt;

    return result;
}

// A mass arriving at the stop with a stated velocity and nothing else acting: the element alone.
struct Entry
{
    double before = 0.0;  // the force on the last tick out of contact
    double first = 0.0;   // the force on the first tick in contact
    double peak = 0.0;
    double impulse = 0.0; // ∫ F dt over the contact, N·s
    double deepest = 0.0; // of `past`, metres
    double rebound = 0.0; // the velocity on leaving, compression-positive (so negative)
    double work = 0.0;    // what the stop took out of the mass, joules
    int ticks = 0;
    bool left = false;
};

[[nodiscard]] Entry enter(const Shaft& shaft, const Law law, const double speed, const double dt)
{
    // Start half a tick short of the gap so the second step lands inside it: the whole point is
    // the first force after contact.
    auto compression = shaft.stop.gap - speed * dt * 0.5;
    auto velocity = speed;
    auto result = Entry{};
    auto wasIn = false;

    for (auto tick = 0; tick < 200000; tick++)
    {
        const auto in = compression - shaft.stop.gap > 0.0;
        if (wasIn && !in)
        {
            result.rebound = velocity;
            result.left = true;
            break;
        }

        const auto next = advance(shaft, Scenario{}, law, dt, compression, velocity);

        if (in)
        {
            if (!wasIn)
            {
                result.first = next.stopForce;
            }
            result.peak = std::max(result.peak, next.stopForce);
            result.impulse += next.stopForce * dt;
            result.deepest = std::max(result.deepest, compression - shaft.stop.gap);
            result.ticks++;
        }
        else
        {
            result.before = next.stopForce;
        }

        // The work-energy identity of the recurrence itself: `m (v' − v) = −F dt` exactly, so the
        // kinetic energy the stop took is `F · ½ (v + v') · dt`, tick by tick.
        result.work += next.stopForce * 0.5 * (velocity + next.velocity) * dt;

        wasIn = in;
        compression = next.compression;
        velocity = next.velocity;
    }

    return result;
}

} // namespace

TEST_CASE("the stop's zero-velocity curve is the authored one to the bit, and its entry is continuous",
          "[physics][suspension][stop]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        for (const auto bump : {true, false})
        {
            const auto& corner = golf->corners[index];
            const auto& stop = bump ? corner.bumpStop : corner.droopStop;
            INFO((index == 0 ? "front " : "rear ") << (bump ? "bump" : "droop"));

            // R3: the elastic law, written as it always was, against the element at rest. Bit
            // equality, not a tolerance.
            for (auto slice = 1; slice <= 2000; slice++)
            {
                const auto past = 0.040 * static_cast<double>(slice) / 2000.0;
                const auto authored = stop.rate * std::pow(past, stop.progression) /
                                      std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);

                REQUIRE(stop.elasticForce(past) == authored);
                REQUIRE(stop.force(past, 0.0) == authored);
                REQUIRE(stop.force(past) == authored);
                REQUIRE(stop.terms(past, 0.0).explicitForce == authored);
            }

            // R1: nothing outside the stop, at any velocity, and no coefficient either.
            for (const auto velocity : {-1.0, -0.1, 0.0, 0.1, 1.0})
            {
                REQUIRE(stop.force(0.0, velocity) == 0.0);
                REQUIRE(stop.force(-0.001, velocity) == 0.0);
                REQUIRE(stop.terms(-0.001, velocity).viscousCoefficient == 0.0);
                REQUIRE(stop.terms(-0.001, velocity).explicitForce == 0.0);
                REQUIRE(stop.viscousCoefficient(0.0) == 0.0);
            }

            // R2: the total force goes to zero with the compression at any velocity, and its slope
            // does too — the viscous share is `damping · (x/gap)^(p−1) · v`, quadratic in x on a
            // cubic stop, so the force is C1 at the touch point.
            for (const auto velocity : {0.05, 0.15, 0.30, 0.70, 1.5})
            {
                const auto fine = stop.force(1e-6, velocity);
                const auto coarse = stop.force(1e-3, velocity);
                REQUIRE(fine < 1e-3);
                REQUIRE(fine / 1e-6 < coarse / 1e-3);
                REQUIRE(stop.force(2e-4, velocity) / stop.force(1e-4, velocity) > 3.9);
            }

            // The coefficient is the authored constant at one gap, by construction.
            REQUIRE(stop.viscousCoefficient(stop.gap) == Catch::Approx(stop.damping).epsilon(1e-12));

            // R4 at the law: the viscous share always opposes the stop-relative velocity.
            for (const auto past : {0.001, 0.005, 0.02})
            {
                for (const auto velocity : {-0.7, -0.05, 0.05, 0.7})
                {
                    const auto viscous = stop.terms(past, velocity).viscousCoefficient * velocity;
                    REQUIRE(viscous * velocity >= 0.0);
                }
            }
        }
    }
}

TEST_CASE("a mass entering the stop meets a force that rises from zero", "[physics][suspension][stop]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    constexpr auto dt = 1.0 / 360.0;
    const auto speeds = std::array{0.05, 0.15, 0.30, 0.70};

    std::printf("\n=== finite-velocity entry: the shaft's mass and its bump stop alone, 360 Hz ===\n");
    std::printf("  before = force on the last tick out of contact; first = the first tick in; work = what the stop "
                "took out.\n");

    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        const auto shaft = shaftOf(golf->corners[index], index == 0 ? "front" : "rear");

        std::printf("\n  --- %s: shaft mass %.2f kg, stop gap %.1f mm, rate %.0f, p %.1f, damping %.0f N.s/m ---\n",
                    shaft.name, shaft.mass, 1000.0 * shaft.stop.gap, shaft.stop.rate, shaft.stop.progression,
                    shaft.stop.damping);
        std::printf("  %5s %4s %8s %8s %8s %9s %8s %8s %8s %6s\n", "v m/s", "law", "before N", "first N", "peak N",
                    "impulse", "deep mm", "out m/s", "work J", "ticks");

        for (const auto speed : speeds)
        {
            const auto in = 0.5 * shaft.mass * speed * speed;
            auto entries = std::array<Entry, 3>{};
            for (const auto law : {Law::Old, Law::New, Law::Zero})
            {
                const auto entry = enter(shaft, law, speed, dt);
                entries[static_cast<std::size_t>(law)] = entry;
                std::printf("  %5.2f %4s %8.1f %8.1f %8.1f %9.4f %8.2f %8.3f %8.2f %6d\n", speed, lawName(law),
                            entry.before, entry.first, entry.peak, entry.impulse, 1000.0 * entry.deepest, entry.rebound,
                            entry.work, entry.ticks);
            }

            const auto& oldLaw = entries[0];
            const auto& newLaw = entries[1];

            // R1 both ways: nothing before contact.
            REQUIRE(oldLaw.before == 0.0);
            REQUIRE(newLaw.before == 0.0);

            // D2, the witness: the old law's first force is the constant times the velocity —
            // 28 kN at 0.7 m/s on the front — and the new law's is the law evaluated at the first
            // tick's own compression, tens of newtons at most.
            REQUIRE(oldLaw.first > 0.9 * shaft.stop.damping * speed);
            REQUIRE(newLaw.first < 0.02 * oldLaw.first);
            REQUIRE(newLaw.first <= shaft.stop.force(speed * dt, speed) * 1.001 + 1e-9);

            // The peak arrives after the first tick, from a force that grew into the stop.
            REQUIRE(newLaw.peak > newLaw.first);
            REQUIRE(newLaw.left);

            // R4 on the event: the mass leaves slower than it arrived, by exactly what the stop took.
            REQUIRE(newLaw.rebound < 0.0);
            REQUIRE(-newLaw.rebound < speed);
            REQUIRE(newLaw.work > 0.0);
            REQUIRE(newLaw.work ==
                    Catch::Approx(in - 0.5 * shaft.mass * newLaw.rebound * newLaw.rebound).epsilon(1e-8));
        }
    }
}

TEST_CASE("a corner resting on its stop settles instead of alternating at the tick rate", "[physics][suspension][stop]")
{
    // D1, reproduced: the characterisation's 400 kg over the front axle — 200 kg per corner referred
    // onto the shaft — resting on the front stop with the corner's own spring, damper and tyre
    // acting, given a 30 mm/s kick (the car's own cycle ran at ±24.5 mm/s). The old law locks into
    // the force-on / force-off alternation at half the tick rate and never leaves it; the new one
    // converges.
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    constexpr auto dt = 1.0 / 360.0;
    const auto& front = golf->corners[0];
    const auto shaft = shaftOf(front, "front");

    // The spring carries the static corner load at zero compression, so the load into the stop is
    // the added mass alone, referred through the wheel-to-shaft ratio. The rest position is where
    // the spring and the elastic law carry it together, by bisection; the tyre is referenced there.
    auto scenario = Scenario{.spring = true,
                             .damper = true,
                             .load = 200.0 * gravity * shaft.wheelPerShaft,
                             .tyreRate = front.tireVerticalRate * shaft.wheelPerShaft * shaft.wheelPerShaft,
                             .tyreDamping = front.tireVerticalDamping * shaft.wheelPerShaft * shaft.wheelPerShaft};
    {
        auto low = shaft.stop.gap;
        auto high = shaft.stop.gap + 0.05;
        for (auto iteration = 0; iteration < 100; iteration++)
        {
            const auto middle = 0.5 * (low + high);
            (shaft.springRate * middle + shaft.stop.elasticForce(middle - shaft.stop.gap) < scenario.load ? low : high) =
                middle;
        }
        scenario.equilibrium = 0.5 * (low + high);
    }
    constexpr auto kick = 0.03;

    struct Rest
    {
        int zeroTicks = 0;      // ticks in contact with no force at all
        int signChanges = 0;    // of the velocity, over the last second
        double forceMin = 1e9;
        double forceMax = 0.0;
        double velocityRms = 0.0;
        double compression = 0.0;
        double frequency = 0.0; // of the force's deviation from its mean, by zero crossings, Hz
    };

    const auto rest = [&](const Law law)
    {
        auto compression = scenario.equilibrium;
        auto velocity = kick;
        auto result = Rest{};
        auto forces = std::vector<double>{};
        auto velocities = std::vector<double>{};

        constexpr auto ticks = 4 * 360;
        for (auto tick = 0; tick < ticks; tick++)
        {
            const auto next = advance(shaft, scenario, law, dt, compression, velocity);
            if (tick >= ticks - 360)
            {
                forces.push_back(next.stopForce);
                velocities.push_back(next.velocity);
                if (compression - shaft.stop.gap > 0.0 && next.stopForce == 0.0)
                {
                    result.zeroTicks++;
                }
                result.forceMin = std::min(result.forceMin, next.stopForce);
                result.forceMax = std::max(result.forceMax, next.stopForce);
            }
            compression = next.compression;
            velocity = next.velocity;
        }

        auto sum = 0.0;
        for (const auto v : velocities)
        {
            sum += v * v;
        }
        result.velocityRms = std::sqrt(sum / static_cast<double>(velocities.size()));
        for (auto index = std::size_t{1}; index < velocities.size(); index++)
        {
            result.signChanges += (velocities[index] > 0.0) != (velocities[index - 1] > 0.0) ? 1 : 0;
        }
        auto mean = 0.0;
        for (const auto f : forces)
        {
            mean += f;
        }
        mean /= static_cast<double>(forces.size());
        auto crossings = 0;
        for (auto index = std::size_t{1}; index < forces.size(); index++)
        {
            crossings += ((forces[index] - mean) > 0.0) != ((forces[index - 1] - mean) > 0.0) ? 1 : 0;
        }
        result.frequency = 0.5 * static_cast<double>(crossings);
        result.compression = compression;

        return result;
    };

    std::printf("\n=== resting on the front stop: 200 kg per corner, a %.0f mm/s kick, 4 s, the last second read ===\n",
                1000.0 * kick);
    std::printf("  shaft mass %.2f kg, shaft spring %.0f N/m, tyre %.0f N/m + %.0f N.s/m, load %.0f N, rest %.3f mm past "
                "(elastic %.1f N); old alpha = c dt / m = %.3f, the tyre's %.3f\n",
                shaft.mass, shaft.springRate, scenario.tyreRate, scenario.tyreDamping, scenario.load,
                1000.0 * (scenario.equilibrium - shaft.stop.gap),
                shaft.stop.elasticForce(scenario.equilibrium - shaft.stop.gap), shaft.stop.damping * dt / shaft.mass,
                scenario.tyreDamping * dt / shaft.mass);
    std::printf("  %4s %9s %9s %9s %12s %11s %10s %8s\n", "law", "zero tks", "sign chg", "Fmin N", "Fmax N",
                "v rms m/s", "past mm", "f Hz");

    auto results = std::array<Rest, 3>{};
    for (const auto law : {Law::Old, Law::New, Law::Zero})
    {
        const auto r = rest(law);
        results[static_cast<std::size_t>(law)] = r;
        std::printf("  %4s %9d %9d %9.1f %12.1f %11.2e %10.3f %8.1f\n", lawName(law), r.zeroTicks, r.signChanges,
                    r.forceMin, r.forceMax, r.velocityRms, 1000.0 * (r.compression - shaft.stop.gap), r.frequency);
    }

    const auto& oldLaw = results[0];
    const auto& newLaw = results[1];

    // The replica reproduces the defect: the old law is still alternating in its fourth second,
    // with the force off on a large share of the ticks and the velocity changing sign at close to
    // the tick rate. If this stops holding the replica has drifted from the corner, not the corner
    // from the defect.
    CHECK(oldLaw.zeroTicks > 100);
    CHECK(oldLaw.signChanges > 200);
    CHECK(oldLaw.frequency > 100.0);

    // R5: the new law has converged — no zero-force tick on the stop, a velocity at numerical rest,
    // and a force that is one number.
    REQUIRE(newLaw.zeroTicks == 0);
    REQUIRE(newLaw.velocityRms < 1e-7);
    REQUIRE(newLaw.forceMax - newLaw.forceMin < 1e-3);
    REQUIRE(newLaw.signChanges < 3);

    // And it rests where the elastic law and the load say: the same depth the zero-damping control
    // reaches, because at rest the dynamic term is nothing.
    REQUIRE(newLaw.compression == Catch::Approx(results[2].compression).epsilon(1e-6));
}

TEST_CASE("the stop's dynamic term takes energy out and never puts it in", "[physics][suspension][stop]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    const auto& front = golf->corners[0].bumpStop;

    // A prescribed 2 Hz stroke to 12 mm past the touch point — the loop `[.jounce-stop]` measured
    // the sourced branch on — with the work split by term: the elastic law returns what it stored
    // (a closed loop), the viscous share is the loss, and the total with the clamp is what the
    // element actually took.
    constexpr auto step = 1.0 / 3600.0;
    constexpr auto frequency = 2.0;
    constexpr auto depth = 0.012;

    struct Loop
    {
        double elastic = 0.0;
        double viscous = 0.0;
        double total = 0.0;
        int clamped = 0;
        double powerMin = 1e9; // min over the cycle of viscous force × velocity
    };

    // Trapezoidal in the displacement, so that a conservative term integrates to zero over the
    // closed path to rounding: the samples going in and coming out sit on the same compressions.
    const auto loop = [&](const bool oldLaw)
    {
        auto result = Loop{};
        auto previousPast = 0.0;
        auto previousElastic = 0.0;
        auto previousViscous = 0.0;
        auto previousTotal = 0.0;
        for (auto index = 1; index <= 1800; index++)
        {
            const auto now = step * static_cast<double>(index);
            const auto phase = 2.0 * std::numbers::pi * frequency * now;
            const auto past = depth * 0.5 * (1.0 - std::cos(phase));
            const auto speed = depth * 0.5 * std::sin(phase) * 2.0 * std::numbers::pi * frequency;
            const auto dx = past - previousPast;

            const auto elastic = front.elasticForce(past);
            const auto viscous =
                oldLaw ? (past > 0.0 ? front.damping * speed : 0.0) : front.viscousCoefficient(past) * speed;
            const auto total = std::max(0.0, elastic + viscous);

            result.elastic += 0.5 * (elastic + previousElastic) * dx;
            result.viscous += 0.5 * (viscous + previousViscous) * dx;
            result.total += 0.5 * (total + previousTotal) * dx;
            result.clamped += (past > 0.0 && total == 0.0) ? 1 : 0;
            result.powerMin = std::min(result.powerMin, viscous * speed);
            previousPast = past;
            previousElastic = elastic;
            previousViscous = viscous;
            previousTotal = total;
        }

        return result;
    };

    const auto before = loop(true);
    const auto after = loop(false);

    std::printf("\n=== a 2 Hz, 12 mm stroke on the front stop: work by term, joules (positive = taken from the motion) "
                "===\n");
    std::printf("  %4s %10s %10s %10s %8s %12s\n", "law", "elastic", "viscous", "total", "clamped", "min F_v.v W");
    std::printf("  %4s %10.3f %10.3f %10.3f %8d %12.3f\n", "old", before.elastic, before.viscous, before.total,
                before.clamped, before.powerMin);
    std::printf("  %4s %10.3f %10.3f %10.3f %8d %12.3f\n", "new", after.elastic, after.viscous, after.total,
                after.clamped, after.powerMin);

    // R4: the elastic law is a closed loop, the viscous share is a loss, the viscous power never
    // changes sign, and the clamped total is a loss too. The total sits a little *below* the
    // viscous share, not above it: where the sum would have pulled on the release stroke the stop
    // lets go instead, and a pull against an outgoing shaft would have taken energy — the clamp
    // forgoes that, it does not add anything.
    REQUIRE(std::abs(after.elastic) < 1e-6);
    REQUIRE(after.viscous > 0.0);
    REQUIRE(after.total > 0.0);
    REQUIRE(after.powerMin >= 0.0);
    REQUIRE(after.total <= after.viscous + 1e-9);
    REQUIRE(after.total > 0.95 * after.viscous);

    // And the same on an integrated event, every tick: the viscous share's power is never negative
    // on the velocity the step arrived at, for every entry speed and both axles, and the stop's
    // work on the mass is positive.
    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        const auto shaft = shaftOf(golf->corners[index], index == 0 ? "front" : "rear");
        for (const auto speed : {0.05, 0.30, 0.70})
        {
            auto compression = shaft.stop.gap - speed / 720.0;
            auto velocity = speed;
            auto work = 0.0;
            for (auto tick = 0; tick < 2000; tick++)
            {
                const auto next = advance(shaft, Scenario{}, Law::New, 1.0 / 360.0, compression, velocity);
                REQUIRE(next.viscous * next.velocity >= 0.0);
                work += next.stopForce * 0.5 * (velocity + next.velocity) / 360.0;
                compression = next.compression;
                velocity = next.velocity;
                if (compression < shaft.stop.gap && velocity < 0.0)
                {
                    break;
                }
            }
            REQUIRE(work > 0.0);
        }
    }
}

TEST_CASE("the stop's entry converges with the timestep", "[physics][suspension][stop]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    const auto rates = std::array{120.0, 240.0, 360.0, 720.0, 1440.0, 2880.0};

    std::printf("\n=== the same entry at every timestep: shaft mass and stop alone ===\n");

    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        const auto shaft = shaftOf(golf->corners[index], index == 0 ? "front" : "rear");

        for (const auto speed : {0.30, 0.70})
        {
            std::printf("\n  --- %s, %.2f m/s ---\n", shaft.name, speed);
            std::printf("  %6s %4s %8s %8s %9s %8s %8s %8s %6s\n", "Hz", "law", "first N", "peak N", "impulse",
                        "deep mm", "out m/s", "work J", "ticks");

            auto reference = Entry{};
            auto coarse = Entry{};
            for (const auto law : {Law::Old, Law::New})
            {
                for (const auto rate : rates)
                {
                    const auto entry = enter(shaft, law, speed, 1.0 / rate);
                    std::printf("  %6.0f %4s %8.1f %8.1f %9.4f %8.2f %8.3f %8.2f %6d\n", rate, lawName(law),
                                entry.first, entry.peak, entry.impulse, 1000.0 * entry.deepest, entry.rebound,
                                entry.work, entry.ticks);
                    if (law == Law::New && rate == 2880.0)
                    {
                        reference = entry;
                    }
                    if (law == Law::New && rate == 360.0)
                    {
                        coarse = entry;
                    }
                }
            }

            // R6: at the engine's own 360 Hz the event is within a few percent of the 2880 Hz
            // answer on every quantity that matters, and the first force after contact vanishes
            // with the timestep rather than staying at the constant's value.
            REQUIRE(coarse.peak == Catch::Approx(reference.peak).epsilon(0.10));
            REQUIRE(coarse.impulse == Catch::Approx(reference.impulse).epsilon(0.10));
            REQUIRE(coarse.deepest == Catch::Approx(reference.deepest).epsilon(0.10));
            REQUIRE(coarse.rebound == Catch::Approx(reference.rebound).epsilon(0.10));
            REQUIRE(coarse.work == Catch::Approx(reference.work).epsilon(0.10));
            REQUIRE(reference.first < coarse.first);
        }
    }
}

TEST_CASE("what the settle's drop does to the stops", "[.stop-element]")
{
    // Every fixture in this project settles the car by dropping it onto a plate from 0.52 m or so,
    // and the scripted launch behind the driving golden drops it too. This prints what that drop
    // does to the four corners' stops in its first two seconds — which is the whole reason a
    // scenario that never touches a stop while it is being recorded is not byte-identical across
    // the 2026-09-08 change: the settle before it was.
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    REQUIRE(raceengine::bringUpJolt().has_value());

    auto descriptor = raceengine::ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};
    const auto world = raceengine::PhysicsWorld::create(raceengine::generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    for (const auto height : {0.52, 0.572})
    {
        auto state = raceengine::VehicleState{};
        state.chassis.position = glm::dvec3(0.0, height, 0.0);

        auto bumpTicks = std::array<int, raceengine::cornerCount>{};
        auto droopTicks = std::array<int, raceengine::cornerCount>{};
        auto bumpPeak = std::array<double, raceengine::cornerCount>{};
        auto droopPeak = std::array<double, raceengine::cornerCount>{};
        auto firstDroop = std::array<double, raceengine::cornerCount>{-1.0, -1.0, -1.0, -1.0};
        auto firstBump = std::array<double, raceengine::cornerCount>{-1.0, -1.0, -1.0, -1.0};
        auto lastAny = -1.0;

        for (auto tick = 0; tick < 720; tick++)
        {
            const auto stepped = raceengine::stepVehicle(golf.value(), state, raceengine::VehicleInput{},
                                                         raceengine::noDriveTorque, world.value(), 1.0 / 360.0);
            REQUIRE(stepped.has_value());
            for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
            {
                const auto& forces = stepped->corners[index].forces;
                const auto now = static_cast<double>(tick) / 360.0;
                if (forces.bumpStop > 0.0)
                {
                    bumpTicks[index]++;
                    bumpPeak[index] = std::max(bumpPeak[index], forces.bumpStop);
                    firstBump[index] = firstBump[index] < 0.0 ? now : firstBump[index];
                    lastAny = now;
                }
                if (forces.droopStop != 0.0)
                {
                    droopTicks[index]++;
                    droopPeak[index] = std::max(droopPeak[index], -forces.droopStop);
                    firstDroop[index] = firstDroop[index] < 0.0 ? now : firstDroop[index];
                    lastAny = now;
                }
            }
        }

        std::printf("\n=== dropped from %.3f m onto a plate, the first two seconds ===\n", height);
        for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
        {
            std::printf("  %s: droop %4d ticks, first at %.3f s, peak %7.1f N | bump %4d ticks, first at %.3f s, peak %7.1f N\n",
                        raceengine::cornerAbbreviation(static_cast<raceengine::Corner>(index)), droopTicks[index],
                        firstDroop[index], droopPeak[index], bumpTicks[index], firstBump[index], bumpPeak[index]);
        }
        std::printf("  last stop contact at %.3f s\n", lastAny);
    }

    raceengine::tearDownJolt();
}
