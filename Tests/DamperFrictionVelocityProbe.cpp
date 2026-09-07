// How big is the damper friction's velocity dependence, and is it safe through zero at 360 Hz?
//
// The shipped law is flat: `friction · tanh(v / 10 mm/s)`, so above about 30 mm/s every corner in
// this project delivers exactly its stated newtons whatever the shaft is doing. The measurement the
// front's 107 N comes from is not flat — Deubel et al.'s steady-state curves for that same Passat B8
// strut rise about 31 % above the quasi-static value near 5 mm/s and then fall to 69 % of it by
// 300 mm/s. This probe sizes that difference and nothing else. **It does not say which is better.**
//
// **Every dynamic fixture here is stop-free by construction and says so.** The bump stop carries a
// placed 40000 N·s/m viscous term that is five times a front corner's critical damping, so a
// measurement that touches it is a measurement of that constant. Every case below drives the corner
// about its design position, where both stops are exactly zero, at an amplitude computed from the
// wanted shaft velocity — and each reports the largest stop force it saw, which must be zero.
//
// Hidden behind a dot tag — `./EngineTests "[.damper-friction-velocity]"`.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::CornerSetup;
using raceengine::damperDampingCoefficient;
using raceengine::damperFrictionReferenceSpeed;
using raceengine::damperFrictionShapeAt;
using raceengine::damperShaftCompression;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::macPhersonStrutFrictionShape;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperForce;
using raceengine::stepVehicle;
using raceengine::SuspensionState;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
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

[[nodiscard]] ProvingGroundDescriptor plate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 200.0;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<raceengine::Feature>{};

    return descriptor;
}

// --- the sign convention, stated once ----------------------------------------------------------
//
// `DamperForceSolution::velocity` is **positive in compression**: the force path takes
// `velocity = −lengthPerAngle · wishboneRate`, and a positive wishbone rate is bump. A positive
// force **resists** compression, so in this model's sign convention a friction force carries the
// same sign as the velocity it opposes.
//
// Two consequences, and getting them the wrong way round is the classic way to publish a damper
// that adds energy. The power the damper delivers **to** the corner is `−F·v`, which is never
// positive. The energy it **removes** is therefore `+∫F·v dt`, and that is what every joule below
// is. (The task this was built for writes the removed energy as `∫−F·v dt`, which is the same
// quantity in the opposite force convention.)

[[nodiscard]] SuspensionState solvedAt(const CornerSetup& corner, const double wishboneAngle)
{
    const auto solved = solveCornerWithJacobian(corner.hardpoints, wishboneAngle, 0.0);
    REQUIRE(solved.has_value());

    return *solved;
}

[[nodiscard]] double jacobianAtDesign(const CornerSetup& corner)
{
    return solveDamperForce(corner, solvedAt(corner, 0.0), 1.0).lengthPerAngle;
}

// What one tick of a corner looks like, with every mechanism this task must keep apart reported
// separately.
struct Sample
{
    double velocity = 0.0;
    double compression = 0.0;
    double viscous = 0.0;
    double friction = 0.0;
    double bumpStop = 0.0;
    double droopStop = 0.0;
};

[[nodiscard]] Sample sampleCorner(const CornerSetup& corner, const double wishboneAngle, const double wishboneRate)
{
    const auto state = solvedAt(corner, wishboneAngle);
    const auto damper = solveDamperForce(corner, state, wishboneRate);
    const auto compression = damperShaftCompression(corner, damper);

    return Sample{.velocity = damper.velocity,
                  .compression = compression,
                  .viscous = corner.damper.at(damper.velocity),
                  .friction = damper.force - corner.damper.at(damper.velocity),
                  .bumpStop = corner.bumpStop.force(compression - corner.bumpStop.gap, damper.velocity),
                  .droopStop = -corner.droopStop.force(-compression - corner.droopStop.gap, -damper.velocity)};
}

// --- the deterministic shaft motions -----------------------------------------------------------
//
// A sinusoid about the design position, sized from the wanted peak shaft velocity rather than from
// a displacement, so each case is named by the thing it is characterising. The displacement that
// falls out is `peak / (2 pi f)` on the shaft, which is what keeps every case inside the free
// travel: at 300 mm/s and 5 Hz that is 9.5 mm against a 20 mm bump gap and a 40 mm droop gap.
struct Motion
{
    std::string name;
    double peakVelocity = 0.0;
    double frequency = 0.0;
    double seconds = 0.0;
    // A decay time constant on the amplitude, seconds. Zero is a steady sinusoid.
    double decay = 0.0;
};

struct Dissipation
{
    double energy = 0.0;
    double peakVelocity = 0.0;
    double peakFriction = 0.0;
    double shaftAmplitude = 0.0;
    double worstStop = 0.0;
    int stopTicks = 0;
};

[[nodiscard]] Dissipation dissipated(const CornerSetup& corner, const Motion& motion)
{
    const auto jacobian = jacobianAtDesign(corner);
    REQUIRE(std::abs(jacobian) > 1e-9);

    const auto omega = 2.0 * std::numbers::pi * motion.frequency;
    // Angle amplitude that puts `peakVelocity` on the shaft at the design Jacobian.
    const auto amplitude = motion.peakVelocity / (omega * std::abs(jacobian));

    auto result = Dissipation{};

    const auto steps = static_cast<int>(std::lround(motion.seconds / tick));
    for (auto step = 0; step < steps; step++)
    {
        const auto time = static_cast<double>(step) * tick;
        const auto envelope = motion.decay > 0.0 ? std::exp(-time / motion.decay) : 1.0;

        const auto angle = amplitude * envelope * std::sin(omega * time);
        const auto rate = amplitude * envelope * omega * std::cos(omega * time);

        const auto sample = sampleCorner(corner, angle, rate);

        result.energy += sample.friction * sample.velocity * tick;
        result.peakVelocity = std::max(result.peakVelocity, std::abs(sample.velocity));
        result.peakFriction = std::max(result.peakFriction, std::abs(sample.friction));
        result.shaftAmplitude = std::max(result.shaftAmplitude, std::abs(sample.compression));

        const auto stop = std::max(std::abs(sample.bumpStop), std::abs(sample.droopStop));
        result.worstStop = std::max(result.worstStop, stop);
        if (stop != 0.0)
        {
            result.stopTicks++;
        }
    }

    return result;
}

} // namespace

TEST_CASE("damper friction against shaft velocity: the two laws, tabulated", "[.damper-friction-velocity]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto& front = built->corners[0];

    auto shaped = front;
    shaped.damperFrictionShape = macPhersonStrutFrictionShape();

    const auto jacobian = jacobianAtDesign(front);

    WARN("=== front corner at its design position, where both travel stops are exactly zero ===");
    WARN("sign convention: velocity POSITIVE in compression; a friction force carries the SIGN of "
         "the velocity it opposes; power delivered to the corner is -F*v and is never positive; "
         "energy removed is +F*v integrated");
    WARN("magnitude " << front.damperFriction << " N (quasi-static sliding, Deubel et al. pin slider), "
                      << "regularisation width " << front.damperFrictionSpeed * 1000.0 << " mm/s, "
                      << "shape normalised at " << damperFrictionReferenceSpeed * 1000.0 << " mm/s");
    WARN("   v mm/s |  current N | candidate N | cand/mag | shape | current W dissipated | candidate W |  diff N");

    for (const auto millimetres : {-500.0, -300.0, -200.0, -100.0, -50.0, -20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0,
                                   20.0, 50.0, 100.0, 200.0, 300.0, 500.0})
    {
        const auto velocity = millimetres / 1000.0;
        const auto rate = -velocity / jacobian;

        const auto current = sampleCorner(front, 0.0, rate);
        const auto candidate = sampleCorner(shaped, 0.0, rate);

        REQUIRE(current.bumpStop == 0.0);
        REQUIRE(current.droopStop == 0.0);

        WARN("  " << millimetres << " | " << current.friction << " | " << candidate.friction << " | "
                  << candidate.friction / front.damperFriction << " | "
                  << damperFrictionShapeAt(shaped.damperFrictionShape, std::abs(velocity)) << " | "
                  << current.friction * velocity << " | " << candidate.friction * velocity << " | "
                  << candidate.friction - current.friction);
    }
}

TEST_CASE("damper friction against the viscous damper it sits on", "[.damper-friction-velocity]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    WARN("=== friction against viscous force, both corners, stop force zero throughout ===");
    WARN("axle |  v mm/s | friction N | viscous N | ratio | candidate N | cand ratio");

    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        const auto& corner = built->corners[index];
        auto shaped = corner;
        shaped.damperFrictionShape = macPhersonStrutFrictionShape();

        const auto jacobian = jacobianAtDesign(corner);

        for (const auto millimetres : {1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0})
        {
            const auto velocity = millimetres / 1000.0;
            const auto rate = -velocity / jacobian;

            const auto current = sampleCorner(corner, 0.0, rate);
            const auto candidate = sampleCorner(shaped, 0.0, rate);

            REQUIRE(current.bumpStop == 0.0);
            REQUIRE(current.droopStop == 0.0);

            WARN((index == 0 ? "front" : "rear ")
                 << " | " << millimetres << " | " << current.friction << " | " << current.viscous << " | "
                 << current.friction / current.viscous << " | " << candidate.friction << " | "
                 << candidate.friction / current.viscous);
        }
    }
}

TEST_CASE("damper friction energy over stop-free shaft motions", "[.damper-friction-velocity]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto& front = built->corners[0];

    auto shaped = front;
    shaped.damperFrictionShape = macPhersonStrutFrictionShape();

    const auto motions = std::array{
        Motion{
            .name = "A very low speed (2 mm/s, 0.5 Hz, 8 s)", .peakVelocity = 0.002, .frequency = 0.5, .seconds = 8.0},
        Motion{.name = "B across the sourced peak (20 mm/s, 1 Hz, 6 s)",
               .peakVelocity = 0.020,
               .frequency = 1.0,
               .seconds = 6.0},
        Motion{.name = "C medium (75 mm/s, 2 Hz, 4 s)", .peakVelocity = 0.075, .frequency = 2.0, .seconds = 4.0},
        Motion{.name = "D high (300 mm/s, 5 Hz, 4 s)", .peakVelocity = 0.300, .frequency = 5.0, .seconds = 4.0},
        Motion{.name = "E decaying (100 mm/s, 2 Hz, tau 1 s, 6 s)",
               .peakVelocity = 0.100,
               .frequency = 2.0,
               .seconds = 6.0,
               .decay = 1.0}};

    WARN("=== energy removed by damper friction alone, stop-free by construction ===");
    WARN("motion | shaft mm | peak mm/s | current J | candidate J | diff J | diff % | worst stop N");

    for (const auto& motion : motions)
    {
        const auto current = dissipated(front, motion);
        const auto candidate = dissipated(shaped, motion);

        // The precondition this whole task turns on: neither law touched a travel stop, so neither
        // figure carries any of the placed 40000 N·s/m.
        CHECK(current.stopTicks == 0);
        CHECK(candidate.stopTicks == 0);
        CHECK(current.worstStop == 0.0);

        // And friction only ever removed energy.
        CHECK(current.energy > 0.0);
        CHECK(candidate.energy > 0.0);

        WARN(motion.name << " | " << current.shaftAmplitude * 1000.0 << " | " << current.peakVelocity * 1000.0 << " | "
                         << current.energy << " | " << candidate.energy << " | " << candidate.energy - current.energy
                         << " | " << (candidate.energy / current.energy - 1.0) * 100.0 << " | " << current.worstStop);
    }
}

TEST_CASE("damper friction through repeated zero crossings at 360 Hz", "[.damper-friction-velocity]")
{
    // The question the `tanh` exists for, asked of the candidate. A friction term that chatters shows
    // up as the force changing sign more often than the velocity does; one that sticks shows up as a
    // corner that stops crossing at all; one that is timestep-dependent shows up as a different
    // answer at half the tick.
    //
    // A third configuration is measured here and is **a numerical experiment and not a proposal**:
    // the candidate with the regularisation narrowed to the source's own quasi-static velocity, which
    // is what it would take for the sourced peak to appear at its measured height rather than half
    // of it. It moves no car and is reported so the cost of that choice is on the record.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto& front = built->corners[0];

    auto shaped = front;
    shaped.damperFrictionShape = macPhersonStrutFrictionShape();

    auto narrow = shaped;
    narrow.damperFrictionSpeed = damperFrictionReferenceSpeed;

    WARN("=== repeated low-amplitude zero crossings, design position, both stops zero ===");
    WARN("law | rate Hz | peak mm/s | velocity sign changes | friction sign changes | worst |dF| per tick N | "
         "max implicit coeff N.s/m");

    for (const auto& [name, corner] : std::array<std::pair<std::string, CornerSetup>, 3>{
             {{"current", front}, {"candidate", shaped}, {"candidate, 0.5 mm/s width", narrow}}})
    {
        for (const auto rateHz : {180.0, 360.0, 720.0})
        {
            const auto step = 1.0 / rateHz;
            const auto jacobian = jacobianAtDesign(corner);
            const auto frequency = 2.0;
            const auto omega = 2.0 * std::numbers::pi * frequency;
            const auto peak = 0.004;
            const auto amplitude = peak / (omega * std::abs(jacobian));

            auto velocitySigns = 0;
            auto frictionSigns = 0;
            auto worstJump = 0.0;
            auto worstCoefficient = 0.0;
            auto previousVelocity = 0.0;
            auto previousFriction = 0.0;
            auto largestVelocity = 0.0;

            const auto steps = static_cast<int>(std::lround(4.0 * rateHz));
            for (auto index = 0; index < steps; index++)
            {
                const auto time = static_cast<double>(index) * step;
                const auto angle = amplitude * std::sin(omega * time);
                const auto rate = amplitude * omega * std::cos(omega * time);

                const auto state = solvedAt(corner, angle);
                const auto damper = solveDamperForce(corner, state, rate);
                const auto friction = damper.force - corner.damper.at(damper.velocity);

                if (index > 0)
                {
                    if (damper.velocity * previousVelocity < 0.0)
                    {
                        velocitySigns++;
                    }

                    if (friction * previousFriction < 0.0)
                    {
                        frictionSigns++;
                    }

                    worstJump = std::max(worstJump, std::abs(friction - previousFriction));
                }

                worstCoefficient = std::max(worstCoefficient, damperDampingCoefficient(corner, damper));
                largestVelocity = std::max(largestVelocity, std::abs(damper.velocity));

                previousVelocity = damper.velocity;
                previousFriction = friction;

                const auto compression = damperShaftCompression(corner, damper);
                REQUIRE(corner.bumpStop.force(compression - corner.bumpStop.gap, damper.velocity) == 0.0);
            }

            // The chatter test, and it is an identity rather than a threshold: a friction force that
            // opposes the motion changes sign exactly when the velocity does, so any excess is the
            // term flipping on its own.
            CHECK(frictionSigns == velocitySigns);

            WARN(name << " | " << rateHz << " | " << largestVelocity * 1000.0 << " | " << velocitySigns << " | "
                      << frictionSigns << " | " << worstJump << " | " << worstCoefficient);
        }
    }
}

TEST_CASE("a whole corner ringing down, with the stop count beside it", "[.damper-friction-velocity]")
{
    // The offline A/B, and its primary result is valid only with the stop engagement count at zero.
    // The impulse is deliberately gentler than `[.damper-friction]`'s 1.5 m/s for exactly that
    // reason: that probe measures the settle a seat report named and does not care which mechanism
    // stopped the corner, and this one may not touch the 40000 N·s/m at all.
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    struct Ring
    {
        double settlingTime = 0.0;
        double peak = 0.0;
        double frictionEnergy = 0.0;
        double viscousEnergy = 0.0;
        double loadSpread = 0.0;
        double settledCompression = 0.0;
        double peakCompression = 0.0;
        double peakVelocity = 0.0;
        int stopTicks = 0;
    };

    const auto ring = [&world](const VehicleSetup& setup, const double impulse)
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 100.0);

        for (auto step = 0; step < 2880; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }

        const auto settled = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(settled.has_value());
        const auto rest = settled->corners[0].suspension.wheelTravel;

        state.chassis.linearVelocity = glm::dvec3(0.0, -impulse, 0.0);

        auto result = Ring{};
        auto outside = 0;
        auto lowest = settled->corners[0].forces.tireVertical;
        auto highest = lowest;

        {
            const auto restDamper =
                solveDamperForce(setup.corners[0], settled->corners[0].suspension, state.corners[0].wishboneRate);
            result.settledCompression = damperShaftCompression(setup.corners[0], restDamper);
        }

        constexpr auto steps = 360 * 6;
        for (auto step = 1; step <= steps; step++)
        {
            const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto& solution = stepped->corners[0];
            const auto& corner = setup.corners[0];

            const auto viscous = corner.damper.at(solution.damperVelocity);
            const auto friction = solution.forces.damper - viscous;

            result.frictionEnergy += friction * solution.damperVelocity * tick;
            result.viscousEnergy += viscous * solution.damperVelocity * tick;

            if (solution.forces.bumpStop != 0.0 || solution.forces.droopStop != 0.0)
            {
                result.stopTicks++;
            }

            lowest = std::min(lowest, solution.forces.tireVertical);
            highest = std::max(highest, solution.forces.tireVertical);

            const auto damper = solveDamperForce(corner, solution.suspension, state.corners[0].wishboneRate);
            result.peakCompression = std::max(result.peakCompression, damperShaftCompression(corner, damper));
            result.peakVelocity = std::max(result.peakVelocity, std::abs(solution.damperVelocity));

            const auto excursion = std::abs(solution.suspension.wheelTravel - rest);
            result.peak = std::max(result.peak, excursion * 1000.0);
            if (excursion > 0.002)
            {
                outside = step;
            }
        }

        result.settlingTime = static_cast<double>(outside) * tick;
        result.loadSpread = highest - lowest;

        return result;
    };

    auto candidate = built.value();
    candidate.corners[0].damperFrictionShape = macPhersonStrutFrictionShape();
    candidate.corners[1].damperFrictionShape = macPhersonStrutFrictionShape();

    WARN("=== front corner ring-down, front axle only carries the candidate ===");
    WARN("bump stop gap " << built->corners[0].bumpStop.gap * 1000.0 << " mm on the shaft, droop stop gap "
                          << built->corners[0].droopStop.gap * 1000.0 << " mm");
    WARN("impulse m/s | law | stop ticks | shaft rest/peak mm | peak mm/s | peak travel mm | settles s | "
         "friction J | viscous J | tyre load spread N");

    // **The impulse is scanned rather than chosen, because the validity condition decides it.** A
    // whole corner cannot be told to stay off its stops the way the kinematic fixtures above can, so
    // the fixture reports what each impulse did and the primary result is the largest one that never
    // touched a stop. A number quoted from a run with stop ticks in it is a number about the placed
    // 40000 N·s/m.
    auto primary = -1.0;

    for (const auto impulse : {0.4, 0.3, 0.2, 0.1, 0.05})
    {
        const auto current = ring(built.value(), impulse);
        const auto shaped = ring(candidate, impulse);

        const auto row = [&](const char* name, const Ring& result)
        {
            WARN(impulse << " | " << name << " | " << result.stopTicks << " | " << result.settledCompression * 1000.0
                         << " / " << result.peakCompression * 1000.0 << " | " << result.peakVelocity * 1000.0 << " | "
                         << result.peak << " | " << result.settlingTime << " | " << result.frictionEnergy << " | "
                         << result.viscousEnergy << " | " << result.loadSpread);
        };

        row("current  ", current);
        row("candidate", shaped);
        WARN("   friction energy " << (shaped.frictionEnergy / current.frictionEnergy - 1.0) << " relative, "
                                   << (current.stopTicks == 0 && shaped.stopTicks == 0 ? "STOP-FREE" : "stop-active"));

        if (primary < 0.0 && current.stopTicks == 0 && shaped.stopTicks == 0)
        {
            primary = impulse;
        }
    }

    // The validity condition, asserted rather than hoped for: some impulse in the scan produced a
    // stop-free ring-down at both laws, and that row is the one to read.
    WARN("primary (largest stop-free) impulse: " << primary << " m/s");
    CHECK(primary > 0.0);
}
