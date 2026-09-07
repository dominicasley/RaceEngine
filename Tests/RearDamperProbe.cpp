#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::coaxialSpring;
using raceengine::CornerSetup;
using raceengine::cornerCount;
using raceengine::Curve;
using raceengine::damperElementOf;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::kneedDamper;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::solveDamperGeometry;
using raceengine::solveDamperKinematics;
using raceengine::solveElement;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;

// The **rear damper**, audited: its production curve and every conversion behind it, the braking
// transient it is suspected of driving, the part of its own curve that transient actually visits,
// and a causal A/B that removes it.
// `./EngineTests "[.rear-damper]"`.
//
// Hidden behind a dotted tag like every other probe here: what it produces is a table to read and a
// data question to answer, not a bound to hold. The CHECKs in the last case are invariants rather
// than thresholds — each is a statement that would be a *defect* if it failed, not a number
// somebody chose.
//
// **Why it exists.** The `[.droop-static]` audit closed the droop stop as the owner of the rear
// wheel lift under braking: at minimum rear load there are still ~1.9 mm of shaft extension before
// the 40 mm stop is touched, so nothing about that stop is holding the wheel down when it goes
// light. `AntilockBrakingTests.cpp`'s four-wheels case names the two remaining suspects in its own
// comment — a ~3.3 Hz pitch transient, and the rear damper's rebound knee, *"which came across from
// AC as a wheel-referred knee and has never been checked against a pitch response"*. This probe is
// that check.
//
// **Nothing here changes any car.** Every alternative damper below is built inside a case and
// installed nowhere; production is the shipped arm in every comparison.

namespace
{

constexpr auto tick = 1.0 / 360.0;

// `Vehicle.cppm`'s `earthGravity`, which is not exported. Restated rather than approximated, so the
// unsprung-weight term below is the one the force pass subtracts.
constexpr auto earthGravity = 9.80665;

// The braking fixture's own constants, copied from `AntilockBrakingTests.cpp` so this probe measures
// the scenario that suite measures rather than one of its own invention.
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
constexpr auto startZ = 20.0;

constexpr auto frontLeft = std::size_t{0};
constexpr auto rearLeft = std::size_t{2};
constexpr auto rearRight = std::size_t{3};

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

constexpr auto noBrakePressure = std::array<double, cornerCount>{};

// `AntilockBrakingTests.cpp`'s `gripPlate(1.0, 1.0)`, character for character, so the manoeuvre is
// the one the four-wheels case runs and not one of this probe's invention.
[[nodiscard]] SurfaceMesh gripPlate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(2);
    mesh->materials[0].gripMultiplier = 1.0;
    mesh->materials[0].bumpiness = 0.0;
    mesh->materials[1].gripMultiplier = 1.0;
    mesh->materials[1].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        const auto centroid =
            (mesh->vertices[mesh->indices[triangle * 3 + 0]] + mesh->vertices[mesh->indices[triangle * 3 + 1]] +
             mesh->vertices[mesh->indices[triangle * 3 + 2]]) /
            3.0;

        mesh->surfaces[triangle] = centroid.x < 0.0 ? std::uint32_t{0} : std::uint32_t{1};
    }

    return mesh.value();
}

// --- curve arithmetic, written once so every table reads what the force pass reads ---------------

// The production differencing step and expression order (`VehicleImpl.cpp`'s `curveSlopeAt`).
[[nodiscard]] double curveSlope(const Curve& curve, const double x)
{
    constexpr auto step = 1e-4;

    return (curve.at(x + step) - curve.at(x - step)) / (2.0 * step);
}

// The damper element's design length, which every shaft displacement here is measured against.
[[nodiscard]] double designLengthOf(const CornerSetup& corner)
{
    return solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).length;
}

// --- probe-local damper variants. None of these is installed on any car. -------------------------

// The rear curve with its viscous force deleted outright: a shaft that carries seal friction, the
// spring, both stops and the tyre, and no ordinary damping at all.
[[nodiscard]] Curve deadCurve()
{
    return Curve{.points = {glm::dvec2(-5.0, 0.0), glm::dvec2(0.0, 0.0), glm::dvec2(5.0, 0.0)}};
}

// The shipped curve with one half zeroed and the other left exactly as built. Zeroing a half means
// replacing that side's two y values with zero and leaving both x values where they are, so the
// surviving half is bit-identical to production rather than rebuilt from the rates.
[[nodiscard]] Curve halfCurve(const Curve& shipped, const bool keepBump, const bool keepRebound)
{
    auto curve = shipped;
    REQUIRE(curve.points.size() == 5);

    if (!keepRebound)
    {
        curve.points[0].y = 0.0;
        curve.points[1].y = 0.0;
    }

    if (!keepBump)
    {
        curve.points[3].y = 0.0;
        curve.points[4].y = 0.0;
    }

    return curve;
}

// The shipped curve with the rebound half's forces multiplied. Coarse causal slope only — the knee
// speed does not move, so this scales the force the transient meets and nothing else.
[[nodiscard]] Curve scaledRebound(const Curve& shipped, const double multiplier)
{
    auto curve = shipped;
    REQUIRE(curve.points.size() == 5);

    curve.points[0].y *= multiplier;
    curve.points[1].y *= multiplier;

    return curve;
}

[[nodiscard]] VehicleSetup withRearDamper(const VehicleSetup& base, const Curve& curve)
{
    auto setup = base;
    setup.corners[rearLeft].damper = curve;
    setup.corners[rearRight].damper = curve;

    return setup;
}

// --- the recorded event -------------------------------------------------------------------------

struct Sample
{
    double time = 0.0;
    double speed = 0.0;
    double longitudinalAcceleration = 0.0;
    double pitch = 0.0;
    double pitchRate = 0.0;
    double pitchAcceleration = 0.0;

    std::array<double, cornerCount> load{};
    std::array<double, cornerCount> wheelTravel{};
    std::array<double, cornerCount> q{};
    std::array<double, cornerCount> qdot{};
    // Positive in BUMP throughout, which is `solveDamperForce`'s own sign for the velocity and
    // `damperShaftCompression`'s for the displacement.
    std::array<double, cornerCount> shaftCompression{};
    std::array<double, cornerCount> shaftVelocity{};
    std::array<double, cornerCount> shaftForce{};
    std::array<double, cornerCount> viscous{};
    std::array<double, cornerCount> friction{};
    std::array<double, cornerCount> spring{};
    std::array<double, cornerCount> bumpStop{};
    std::array<double, cornerCount> droopStop{};
    std::array<double, cornerCount> jacobian{};
    std::array<double, cornerCount> travelPerAngle{};
    std::array<double, cornerCount> generalisedSpring{};
    std::array<double, cornerCount> generalisedDamper{};
    std::array<double, cornerCount> generalisedStops{};
    std::array<double, cornerCount> generalisedTyre{};
    std::array<double, cornerCount> generalisedUnsprung{};
    std::array<double, cornerCount> generalisedTotal{};
    std::array<double, cornerCount> slipRatio{};
    std::array<bool, cornerCount> inContact{};
};

struct Run
{
    std::vector<Sample> samples;
    double distance = 0.0;
    bool stopped = false;
    bool onPlate = true;
    // The state the brake phase started from, so an A/B can say the two arms began identically.
    double entrySpeed = 0.0;
    double entryHeight = 0.0;
    std::array<double, cornerCount> entryLoad{};

    [[nodiscard]] std::size_t minimumRearLoadIndex() const
    {
        auto best = std::size_t{0};
        auto lowest = 1e30;

        for (auto index = std::size_t{0}; index < samples.size(); index++)
        {
            const auto load = std::min(samples[index].load[rearLeft], samples[index].load[rearRight]);
            if (load < lowest)
            {
                lowest = load;
                best = index;
            }
        }

        return best;
    }

    [[nodiscard]] double minimumRearLoad() const
    {
        return samples.empty() ? 0.0
                               : std::min(samples[minimumRearLoadIndex()].load[rearLeft],
                                          samples[minimumRearLoadIndex()].load[rearRight]);
    }

    [[nodiscard]] double peakPitch() const
    {
        auto peak = 0.0;
        for (const auto& sample : samples)
        {
            peak = std::max(peak, std::abs(sample.pitch));
        }

        return peak;
    }

    [[nodiscard]] double peakRearExtension() const
    {
        auto peak = 0.0;
        for (const auto& sample : samples)
        {
            peak = std::max(peak, std::max(-sample.shaftCompression[rearLeft], -sample.shaftCompression[rearRight]));
        }

        return peak;
    }

    [[nodiscard]] int airborneRearTicks() const
    {
        auto count = 0;
        for (const auto& sample : samples)
        {
            if (!sample.inContact[rearLeft] || !sample.inContact[rearRight])
            {
                count++;
            }
        }

        return count;
    }

    // **The discriminators the minimum load cannot be.** Once every arm's rear reaches exactly
    // 0 N, a load floor says a wheel lifted and says nothing about when, for how long, or how
    // nearly it was avoided. These three say all of it.
    [[nodiscard]] std::size_t firstLiftIndex() const
    {
        for (auto index = std::size_t{0}; index < samples.size(); index++)
        {
            if (!samples[index].inContact[rearLeft] || !samples[index].inContact[rearRight])
            {
                return index;
            }
        }

        return samples.size();
    }

    [[nodiscard]] double timeToFirstLift() const
    {
        const auto index = firstLiftIndex();

        return index < samples.size() ? samples[index].time : 0.0;
    }

    [[nodiscard]] int firstEpisodeTicks() const
    {
        auto index = firstLiftIndex();
        auto count = 0;

        while (index < samples.size() && (!samples[index].inContact[rearLeft] || !samples[index].inContact[rearRight]))
        {
            count++;
            index++;
        }

        return count;
    }

    [[nodiscard]] int liftEpisodes() const
    {
        auto count = 0;
        auto airborne = false;

        for (const auto& sample : samples)
        {
            const auto off = !sample.inContact[rearLeft] || !sample.inContact[rearRight];
            if (off && !airborne)
            {
                count++;
            }
            airborne = off;
        }

        return count;
    }

    // Newton-seconds of rear axle load lost against the rolling entry load, over the first second.
    // A resolvable, monotone measure of "how light did the rear get and for how long" that does not
    // saturate at zero the way a minimum does.
    [[nodiscard]] double rearLoadDeficit() const
    {
        auto deficit = 0.0;
        const auto reference = entryLoad[rearLeft] + entryLoad[rearRight];

        for (const auto& sample : samples)
        {
            if (sample.time > 1.0)
            {
                break;
            }
            deficit += std::max(0.0, reference - sample.load[rearLeft] - sample.load[rearRight]) * tick;
        }

        return deficit;
    }

    [[nodiscard]] std::vector<double> channel(const std::size_t what, const std::size_t corner) const;
};

// Channel selectors, so the spectral helpers can be handed one signal at a time without a lambda
// per call site. 0 pitch, 1 pitch rate, 2 rear wheel travel, 3 tyre load, 4 shaft velocity,
// 5 damper shaft force.
std::vector<double> Run::channel(const std::size_t what, const std::size_t corner) const
{
    auto values = std::vector<double>{};
    values.reserve(samples.size());

    for (const auto& sample : samples)
    {
        switch (what)
        {
        case 0:
            values.push_back(sample.pitch);
            break;
        case 1:
            values.push_back(sample.pitchRate);
            break;
        case 2:
            values.push_back(sample.wheelTravel[corner]);
            break;
        case 3:
            values.push_back(sample.load[corner]);
            break;
        case 4:
            values.push_back(sample.shaftVelocity[corner]);
            break;
        case 5:
            values.push_back(sample.shaftForce[corner]);
            break;
        default:
            values.push_back(sample.slipRatio[corner]);
            break;
        }
    }

    return values;
}

// The whole braking event, recorded every tick.
//
// **The settle and the roll-in are always run on the SHIPPED car and the arm is installed at the
// instant the pedal moves.** Every arm therefore starts the stop from a bit-identical state, which
// is what makes a one-variable comparison a comparison rather than two different cars measured from
// two different attitudes. The static equilibrium does not depend on the damper — a settled car has
// no shaft velocity — so nothing physical is lost by it; `[.rear-damper]`'s control case switches at
// tick zero of the settle instead and reports whether the answer moves.
[[nodiscard]] Run runBraking(const VehicleSetup& settleSetup, const VehicleSetup& brakeSetup,
                             const PhysicsWorld& world, const double pedal, const bool armAtSettle = false,
                             const double releaseAt = 0.0)
{
    const auto& early = armAtSettle ? brakeSetup : settleSetup;

    auto assists = golfGtiMk7Assists(settleSetup);

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(early, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, hundred);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = hundred / tyreRadius;
    }

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;
        sensors.steeringWheelAngle = lastStep.telemetry.steeringWheelAngle;

        return sensors;
    };

    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped = stepVehicle(early, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    auto run = Run{};
    run.entrySpeed = state.chassis.linearVelocity.z;
    run.entryHeight = state.chassis.position.y;
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        run.entryLoad[index] = lastStep.corners[index].forces.tireVertical;
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
    }

    const auto start = state.chassis.position.z;

    auto input = VehicleInput{};
    input.brake = pedal;

    auto previousPitchRate = 0.0;
    auto elapsed = 0.0;

    for (auto step = 0; step < 360 * 30; step++)
    {
        // The pedal is a step and stays down, exactly as the four-wheels fixture applies it. The one
        // exception is the deliberately-labelled coast control, which releases it at `releaseAt` so
        // the ring-down that follows is a FREE one.
        const auto applied = releaseAt > 0.0 && elapsed >= releaseAt ? 0.0 : pedal;
        input.brake = applied;

        const auto command = updateAssists(assists, assistState, sense(), {.brake = applied, .throttle = 0.0},
                                           brakeCircuitPressures(settleSetup, applied), tick);
        const auto stepped = stepVehicle(brakeSetup, state, input, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        elapsed += tick;

        auto sample = Sample{};
        sample.time = elapsed;
        sample.speed = state.chassis.linearVelocity.z;
        sample.longitudinalAcceleration = lastStep.telemetry.acceleration.z;
        sample.pitch = lastStep.telemetry.pitch;
        sample.pitchRate = lastStep.telemetry.pitchRate;
        sample.pitchAcceleration = (lastStep.telemetry.pitchRate - previousPitchRate) / tick;
        previousPitchRate = lastStep.telemetry.pitchRate;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = brakeSetup.corners[index];
            const auto& solution = lastStep.corners[index];
            const auto geometry = solveDamperGeometry(corner, solution.suspension);
            const auto viscous = corner.damper.at(solution.damperVelocity);
            const auto axisForce =
                solution.forces.spring + solution.forces.damper + solution.forces.bumpStop + solution.forces.droopStop;
            const auto unsprungUpForce = solution.forces.antiRoll - corner.unsprungMass * earthGravity;

            sample.load[index] = solution.forces.tireVertical;
            sample.wheelTravel[index] = solution.suspension.wheelTravel;
            sample.q[index] = state.corners[index].wishboneAngle;
            sample.qdot[index] = state.corners[index].wishboneRate;
            sample.shaftCompression[index] = designLengthOf(corner) - geometry.length;
            sample.shaftVelocity[index] = solution.damperVelocity;
            sample.shaftForce[index] = solution.forces.damper;
            sample.viscous[index] = viscous;
            sample.friction[index] = solution.forces.damper - viscous;
            sample.spring[index] = solution.forces.spring;
            sample.bumpStop[index] = solution.forces.bumpStop;
            sample.droopStop[index] = solution.forces.droopStop;
            sample.jacobian[index] = geometry.lengthPerAngle;
            sample.travelPerAngle[index] = solution.suspension.travelPerAngle;
            sample.generalisedSpring[index] = solution.forces.spring * geometry.lengthPerAngle;
            sample.generalisedDamper[index] = solution.forces.damper * geometry.lengthPerAngle;
            sample.generalisedStops[index] =
                (solution.forces.bumpStop + solution.forces.droopStop) * geometry.lengthPerAngle;
            sample.generalisedUnsprung[index] = unsprungUpForce * solution.suspension.travelPerAngle;
            // The road's own generalised contribution, by subtraction rather than by re-deriving the
            // rotation into the chassis frame: `generalisedForce` is the fused coaxial branch, the
            // shaft-work term is exactly zero with no drive torque, and every other term is above.
            sample.generalisedTyre[index] = solution.generalisedForce - axisForce * geometry.lengthPerAngle -
                                            sample.generalisedUnsprung[index];
            sample.generalisedTotal[index] = solution.generalisedForce;
            sample.slipRatio[index] = lastStep.telemetry.wheels[index].slipRatio;
            sample.inContact[index] = lastStep.telemetry.wheels[index].inContact;
        }

        run.samples.push_back(sample);

        if (std::abs(state.chassis.position.x) > 0.5 * plateWidth - 2.0 || state.chassis.position.z > plateLength - 5.0)
        {
            run.onPlate = false;
        }

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            run.stopped = true;
            break;
        }

        if (releaseAt > 0.0 && elapsed > 4.0)
        {
            break;
        }
    }

    run.distance = state.chassis.position.z - start;

    return run;
}

// --- spectral helpers ---------------------------------------------------------------------------

// A linear detrend over the window, which is what makes a periodogram of a *transient* riding on a
// steady-state shift mean anything: the shift itself is a step, and an undetrended step puts its
// whole energy at the bottom of the band and buys the answer nothing.
[[nodiscard]] std::vector<double> detrended(const std::vector<double>& signal, const std::size_t from,
                                            const std::size_t to)
{
    auto window = std::vector<double>{};
    if (to <= from || to > signal.size())
    {
        return window;
    }

    const auto count = static_cast<double>(to - from);
    auto sumX = 0.0;
    auto sumY = 0.0;
    auto sumXX = 0.0;
    auto sumXY = 0.0;

    for (auto index = from; index < to; index++)
    {
        const auto x = static_cast<double>(index - from);
        sumX += x;
        sumY += signal[index];
        sumXX += x * x;
        sumXY += x * signal[index];
    }

    const auto denominator = count * sumXX - sumX * sumX;
    const auto slope = std::abs(denominator) > 1e-12 ? (count * sumXY - sumX * sumY) / denominator : 0.0;
    const auto intercept = (sumY - slope * sumX) / count;

    window.reserve(to - from);
    for (auto index = from; index < to; index++)
    {
        window.push_back(signal[index] - (intercept + slope * static_cast<double>(index - from)));
    }

    return window;
}

// Magnitude and phase of one frequency, by direct summation. `phase` is the argument of
// `sum x[n] exp(-i 2 pi f n / fs)`, so two signals' phases are comparable and their difference is
// the lead of the first over the second at that frequency.
[[nodiscard]] std::pair<double, double> phasorAt(const std::vector<double>& window, const double frequency)
{
    auto real = 0.0;
    auto imaginary = 0.0;

    for (auto index = std::size_t{0}; index < window.size(); index++)
    {
        const auto angle = 6.283185307179586 * frequency * static_cast<double>(index) * tick;
        real += window[index] * std::cos(angle);
        imaginary -= window[index] * std::sin(angle);
    }

    return {std::hypot(real, imaginary), std::atan2(imaginary, real)};
}

[[nodiscard]] double dominantFrequency(const std::vector<double>& window, const double low, const double high)
{
    auto best = 0.0;
    auto bestPower = -1.0;

    for (auto step = 0; step <= 4000; step++)
    {
        const auto frequency = low + (high - low) * static_cast<double>(step) / 4000.0;
        const auto power = phasorAt(window, frequency).first;

        if (power > bestPower)
        {
            bestPower = power;
            best = frequency;
        }
    }

    return best;
}

// Interior local maxima of a window, as (index, value). Used for cycle-to-cycle period stability and
// for a logarithmic decrement, both of which say something a single periodogram peak cannot.
[[nodiscard]] std::vector<std::pair<std::size_t, double>> localMaxima(const std::vector<double>& window)
{
    auto peaks = std::vector<std::pair<std::size_t, double>>{};

    for (auto index = std::size_t{1}; index + 1 < window.size(); index++)
    {
        if (window[index] > window[index - 1] && window[index] >= window[index + 1])
        {
            peaks.emplace_back(index, window[index]);
        }
    }

    return peaks;
}

// Maxima that are worth calling cycles of an oscillation: above a fraction of the window's own RMS,
// and separated by at least `separation` samples. **The unfiltered version is not usable for a
// period**, and that is a measurement fact rather than a preference — on this transient the raw
// maxima of pitch rate imply frequencies from 2.6 Hz to 180 Hz within one window, because a 360 Hz
// signal riding a nonlinear event has ripples on its peaks.
[[nodiscard]] std::vector<std::pair<std::size_t, double>> prominentMaxima(const std::vector<double>& window,
                                                                          const std::size_t separation)
{
    auto sumSquares = 0.0;
    for (const auto value : window)
    {
        sumSquares += value * value;
    }

    const auto rootMeanSquare = window.empty() ? 0.0 : std::sqrt(sumSquares / static_cast<double>(window.size()));
    const auto threshold = 0.25 * rootMeanSquare;

    auto peaks = std::vector<std::pair<std::size_t, double>>{};

    for (const auto& candidate : localMaxima(window))
    {
        if (candidate.second < threshold)
        {
            continue;
        }

        if (!peaks.empty() && candidate.first - peaks.back().first < separation)
        {
            if (candidate.second > peaks.back().second)
            {
                peaks.back() = candidate;
            }
            continue;
        }

        peaks.push_back(candidate);
    }

    return peaks;
}

struct Axle
{
    const char* name;
    std::size_t index;
};

constexpr auto axles = std::array{Axle{"front", frontLeft}, Axle{"rear", rearLeft}};

// AC's own statement for this car, transcribed from `suspensions.ini` so the conversion below can be
// shown rather than described. Front then rear.
constexpr auto acBumpRate = std::array{4600.0, 6200.0};
constexpr auto acFastBumpRate = std::array{1834.0, 1842.0};
constexpr auto acBumpKnee = std::array{0.070, 0.100};
constexpr auto acReboundRate = std::array{5300.0, 6700.0};
constexpr auto acFastReboundRate = std::array{2589.0, 2700.0};
constexpr auto acReboundKnee = std::array{0.140, 0.140};

} // namespace

TEST_CASE("the shipped rear damper, exactly as the force pass reads it", "[.rear-damper]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    std::printf("\n=== the production damper curve, both axles, as built ===\n");
    std::printf("  A `Curve` is piecewise linear in (shaft velocity [m/s], shaft force [N]).\n");
    std::printf("  Interpolation: linear between neighbouring points (`Curve::at`).\n");
    std::printf("  Extrapolation: NONE. Outside the stated range the END VALUE is held, not the end\n");
    std::printf("  slope — the comment on `Curve::points` says \"the end slope is held\" and the code\n");
    std::printf("  holds `points.front().y` / `points.back().y`. So past +/-5 m/s the damper is a\n");
    std::printf("  CONSTANT force, and its slope there is exactly zero.\n");
    std::printf("  Sign: velocity is POSITIVE IN BUMP (`solveDamperForce`: v = -dL/dq * qdot, and\n");
    std::printf("  dL/dq < 0 on a healthy corner), and force is positive in bump too — the element\n");
    std::printf("  pushing its ends apart. Zero velocity gives exactly zero viscous force, because\n");
    std::printf("  (0,0) is a stated knot on every curve here.\n");

    for (const auto& axle : axles)
    {
        const auto& corner = built->corners[axle.index];
        const auto kinematics = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0, 0.0);
        REQUIRE(kinematics.has_value());

        std::printf("\n  --- %s ---\n", axle.name);
        std::printf("    damper motion ratio at design (dL/dWheelTravel, signed)  %10.6f\n", kinematics->motionRatio);
        std::printf("    damper Jacobian at design      (dL/dq)                   %10.6f m/rad\n",
                    kinematics->lengthPerAngle);
        std::printf("    damper length at design                                  %10.6f m\n", kinematics->length);
        std::printf("    seal friction `damperFriction`                           %10.2f N\n", corner.damperFriction);
        std::printf("    friction regularisation `damperFrictionSpeed`            %10.4f m/s\n",
                    corner.damperFrictionSpeed);
        std::printf("    friction velocity shape stated                           %10s\n",
                    corner.damperFrictionShape.points.empty() ? "no" : "YES");
        std::printf("    spring is coaxial with the damper                        %10s\n",
                    coaxialSpring(corner.hardpoints) ? "YES" : "no");

        std::printf("\n    the curve, %zu knots:\n", corner.damper.points.size());
        std::printf("       v (m/s)      F (N)     branch\n");
        for (const auto& point : corner.damper.points)
        {
            const char* branch = point.x < 0.0 ? "rebound" : (point.x > 0.0 ? "bump" : "origin");
            std::printf("      %8.5f  %10.2f     %s\n", point.x, point.y, branch);
        }

        const auto& knots = corner.damper.points;
        REQUIRE(knots.size() == 5);

        std::printf("\n    rebound knee  %8.5f m/s at %9.2f N   slow slope %10.1f N.s/m\n", -knots[1].x, -knots[1].y,
                    knots[1].y / knots[1].x);
        std::printf("    bump    knee  %8.5f m/s at %9.2f N   slow slope %10.1f N.s/m\n", knots[3].x, knots[3].y,
                    knots[3].y / knots[3].x);
        std::printf("    rebound fast slope %10.1f N.s/m      bump fast slope %10.1f N.s/m\n",
                    (knots[0].y - knots[1].y) / (knots[0].x - knots[1].x),
                    (knots[4].y - knots[3].y) / (knots[4].x - knots[3].x));
        std::printf("    force at +/-5 m/s (the held ends)  bump %9.2f N   rebound %9.2f N\n", knots[4].y, -knots[0].y);
        std::printf("    ASYMMETRY rebound/bump: slow %.4f, fast %.4f, knee speed %.4f\n",
                    (knots[1].y / knots[1].x) / (knots[3].y / knots[3].x),
                    ((knots[0].y - knots[1].y) / (knots[0].x - knots[1].x)) /
                        ((knots[4].y - knots[3].y) / (knots[4].x - knots[3].x)),
                    -knots[1].x / knots[3].x);

        std::printf("\n       v (m/s)   viscous F (N)   dF/dv (N.s/m)   +friction (N)   total (N)\n");
        for (const auto velocity : {-1.0, -0.5, -0.30, -0.20, -0.1382, -0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.0987,
                                    0.20, 0.50, 1.0})
        {
            const auto viscous = corner.damper.at(velocity);
            const auto friction =
                corner.damperFriction * std::tanh(velocity / std::max(corner.damperFrictionSpeed, 1e-9));
            std::printf("      %8.4f   %13.2f   %13.1f   %13.2f   %9.2f\n", velocity, viscous,
                        curveSlope(corner.damper, velocity), friction, viscous + friction);
        }
    }

    std::printf("\n=== the force path, with signs and SI units at every stage ===\n");
    std::printf("  1. wheel motion            wheelTravel [m], up positive              (SuspensionState)\n");
    std::printf("  2. generalised coordinate  q = wishboneAngle [rad], POSITIVE IN BUMP  (VehicleState)\n");
    std::printf("     dWheelTravel/dq = travelPerAngle [m/rad], POSITIVE on this car\n");
    std::printf("  3. generalised rate        qdot = wishboneRate [rad/s], positive COMPRESSING\n");
    std::printf("  4. shaft velocity          v = -(dL/dq) * qdot  [m/s], POSITIVE IN BUMP\n");
    std::printf("     dL/dq = damper element's lengthPerAngle [m/rad], NEGATIVE on this car\n");
    std::printf("  5. Curve::at(v)            viscous shaft force [N], positive in bump\n");
    std::printf("  6. seal friction           + damperFriction * tanh(v / damperFrictionSpeed)  [N]\n");
    std::printf("     (branched: a corner with damperFriction <= 0 runs step 5 alone)\n");
    std::printf("  7. shaft force             forces.damper [N], positive = element pushing apart\n");
    std::printf("  8. damper Jacobian         Q_damper = forces.damper * (dL/dq)  [N.m/rad]\n");
    std::printf("     so a POSITIVE (bump) shaft force gives a NEGATIVE generalised force, which\n");
    std::printf("     drives q down — i.e. extends the corner. That is the correct sense.\n");
    std::printf("  9. generalised force       Q = (spring + damper + stops) * dL/dq\n");
    std::printf("                               + dot(tyreForce_body, patchPerAngle)\n");
    std::printf("                               + (antiRoll - m_u*g) * travelPerAngle\n");
    std::printf("                               + shaftWork      [N.m/rad]\n");
    std::printf("     (fused coaxial branch; shaftWork is exactly 0 with no drive torque)\n");
    std::printf(" 10. integration            qddot = Q / (m_u * |dC/dq|^2), then the damper's own\n");
    std::printf("     slope is solved IMPLICITLY: qdot /= 1 + (c_q/m_q)*dt with\n");
    std::printf("     c_q = (dL/dq)^2 * max(0, dF/dv + friction slope)  [N.m.s/rad]\n");
    std::printf("\n  NOTE the implicit step: the damper's force enters TWICE — explicitly through Q\n");
    std::printf("  at the tick's start velocity, and implicitly as a rate divisor. That is the same\n");
    std::printf("  arithmetic for every arm below, so it does not confound the A/B.\n");
}

TEST_CASE("what Assetto Corsa supplied, and every conversion applied to it", "[.rear-damper]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    std::printf("\n=== SOURCE FACT: suspensions.ini, verbatim ===\n");
    std::printf("  [FRONT] TYPE=STRUT   DAMP_BUMP=4600  DAMP_FAST_BUMP=1834  DAMP_FAST_BUMPTHRESHOLD=0.070\n");
    std::printf("                       DAMP_REBOUND=5300 DAMP_FAST_REBOUND=2589 DAMP_FAST_REBOUNDTHRESHOLD=0.140\n");
    std::printf("  [REAR]  TYPE=DWB     DAMP_BUMP=6200  DAMP_FAST_BUMP=1842  DAMP_FAST_BUMPTHRESHOLD=0.100\n");
    std::printf("                       DAMP_REBOUND=6700 DAMP_FAST_REBOUND=2700 DAMP_FAST_REBOUNDTHRESHOLD=0.140\n");
    std::printf("  setup.ini states NO damper adjustment of any kind, so those six numbers per axle\n");
    std::printf("  are the file's whole statement about damping.\n");

    std::printf("\n=== the convention question, decided from the DATA and not from the names ===\n");
    std::printf("  AC's [REAR] section states, in full: WBCAR_TOP_FRONT, WBCAR_TOP_REAR,\n");
    std::printf("  WBCAR_BOTTOM_FRONT, WBCAR_BOTTOM_REAR, WBTYRE_TOP, WBTYRE_BOTTOM, WBCAR_STEER,\n");
    std::printf("  WBTYRE_STEER, ROD_LENGTH. **There is no damper hardpoint of any kind.**\n");
    std::printf("  A shaft-referred coefficient is only meaningful beside a shaft, and the file\n");
    std::printf("  states none for this axle — so on the rear there is no motion ratio AC could have\n");
    std::printf("  divided by, and DAMP_* can only be AT THE WHEEL. Same argument for SPRING_RATE,\n");
    std::printf("  which this model already reads that way (`corner.springRate = wheelRate / MR^2`).\n");
    std::printf("  Front is weaker evidence in the same direction: the strut axis IS stated\n");
    std::printf("  (STRUT_CAR/STRUT_TYRE), so AC could refer a front figure to a shaft; but the two\n");
    std::printf("  axles' keys are the same keys in the same file, and a file that cannot mean\n");
    std::printf("  shaft-referred on one axle is not stating shaft-referred on the other.\n");
    std::printf("\n  VERDICT: **wheel-referred by elimination.** This is inference from the absence of\n");
    std::printf("  geometry, not a documented convention — no AC documentation is on this machine and\n");
    std::printf("  none was consulted. Classify as SOURCE-DERIVED WITH AMBIGUOUS CONVENTION.\n");

    std::printf("\n=== DERIVED RESULT: the conversion `kneedDamper` applies, arithmetic in full ===\n");
    std::printf("  r  = |damper motion ratio| at design (dL/dWheelTravel)\n");
    std::printf("  knee speed  v_k = r * knee              (wheel m/s -> shaft m/s)\n");
    std::printf("  knee force  F_k = (rate / r^2) * v_k    (wheel N.s/m -> shaft N.s/m)\n");
    std::printf("  end   force F_5 = F_k + (fast / r^2) * (5 - v_k)\n");
    std::printf("  The pair is exact for power: a wheel-referred c gives shaft c/r^2 at r times the\n");
    std::printf("  speed, so the WHEEL-REFERRED coefficient round-trips to c whatever r is. That is\n");
    std::printf("  what makes the placed rear ratio nearly free for the viscous damper — and it is\n");
    std::printf("  NOT free for the knee, which lands at r * knee in shaft units.\n");

    for (auto axle = std::size_t{0}; axle < 2; axle++)
    {
        const auto index = axles[axle].index;
        const auto& corner = built->corners[index];
        const auto kinematics = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0, 0.0);
        REQUIRE(kinematics.has_value());

        const auto ratio = std::abs(kinematics->motionRatio);
        const auto square = ratio * ratio;

        const auto bumpKneeSpeed = ratio * acBumpKnee[axle];
        const auto bumpKneeForce = (acBumpRate[axle] / square) * bumpKneeSpeed;
        const auto bumpEnd = bumpKneeForce + (acFastBumpRate[axle] / square) * (5.0 - bumpKneeSpeed);
        const auto reboundKneeSpeed = ratio * acReboundKnee[axle];
        const auto reboundKneeForce = (acReboundRate[axle] / square) * reboundKneeSpeed;
        const auto reboundEnd = reboundKneeForce + (acFastReboundRate[axle] / square) * (5.0 - reboundKneeSpeed);

        std::printf("\n  --- %s, r = %.6f, r^2 = %.6f ---\n", axles[axle].name, ratio, square);
        std::printf("    bump    slow  %7.1f / r^2 = %9.1f N.s/m at the shaft\n", acBumpRate[axle],
                    acBumpRate[axle] / square);
        std::printf("    bump    fast  %7.1f / r^2 = %9.1f N.s/m\n", acFastBumpRate[axle],
                    acFastBumpRate[axle] / square);
        std::printf("    bump    knee  %7.4f * r   = %9.6f m/s  ->  %9.2f N\n", acBumpKnee[axle], bumpKneeSpeed,
                    bumpKneeForce);
        std::printf("    rebound slow  %7.1f / r^2 = %9.1f N.s/m\n", acReboundRate[axle],
                    acReboundRate[axle] / square);
        std::printf("    rebound fast  %7.1f / r^2 = %9.1f N.s/m\n", acFastReboundRate[axle],
                    acFastReboundRate[axle] / square);
        std::printf("    rebound knee  %7.4f * r   = %9.6f m/s  ->  %9.2f N\n", acReboundKnee[axle], reboundKneeSpeed,
                    reboundKneeForce);
        std::printf("    held ends     bump %9.2f N        rebound %9.2f N\n", bumpEnd, reboundEnd);

        // The five knots the conversion produces, against the five the car carries. Same arithmetic,
        // so this is a transcription check on this probe and not a claim about the car.
        const auto& knots = corner.damper.points;
        REQUIRE(knots.size() == 5);
        CHECK(std::abs(knots[0].x + 5.0) < 1e-12);
        CHECK(std::abs(knots[0].y + reboundEnd) < 1e-6);
        CHECK(std::abs(knots[1].x + reboundKneeSpeed) < 1e-12);
        CHECK(std::abs(knots[1].y + reboundKneeForce) < 1e-9);
        CHECK(std::abs(knots[2].x) < 1e-15);
        CHECK(std::abs(knots[2].y) < 1e-15);
        CHECK(std::abs(knots[3].x - bumpKneeSpeed) < 1e-12);
        CHECK(std::abs(knots[3].y - bumpKneeForce) < 1e-9);
        CHECK(std::abs(knots[4].x - 5.0) < 1e-12);
        CHECK(std::abs(knots[4].y - bumpEnd) < 1e-6);

        std::printf("    round trip: wheel-referred slow rebound = shaft slope * r^2 = %9.2f N.s/m\n",
                    (acReboundRate[axle] / square) * square);
        std::printf("    so the WHEEL sees exactly AC's %.0f N.s/m whatever r is.\n", acReboundRate[axle]);
    }

    std::printf("\n=== what the conversion is NOT ===\n");
    std::printf("  It is not a dimensionless multiplier, it is not already ratio-corrected in the\n");
    std::printf("  file, and nothing about the AC numbers has been rescaled, curve-fitted or tuned:\n");
    std::printf("  `PublishedCarsImpl.cpp` states them as six literals per axle and hands them\n");
    std::printf("  straight to `kneedDamper`. The ONLY free choices this project made are (a) that\n");
    std::printf("  they are wheel-referred and (b) the rear damper's motion ratio, which is placed\n");
    std::printf("  geometry rather than sourced. Nothing else stands between the file and the car.\n");
}

namespace
{

void reportBody(const Run& run, const double until, const int every)
{
    std::printf("\n    t (s)   speed   a_z      pitch     pitchRate  pitchAcc |   FL      FR      RL      RR  |"
                "  front    rear   total   transfer\n");
    std::printf("            (m/s)  (m/s2)    (mrad)     (rad/s)   (rad/s2) |  (N)     (N)     (N)     (N)  |"
                "   (N)     (N)     (N)      (N)\n");

    for (auto index = std::size_t{0}; index < run.samples.size(); index++)
    {
        const auto& sample = run.samples[index];
        if (sample.time > until)
        {
            break;
        }
        if (static_cast<int>(index) % every != 0)
        {
            continue;
        }

        const auto front = sample.load[0] + sample.load[1];
        const auto rear = sample.load[2] + sample.load[3];
        const auto entryFront = run.entryLoad[0] + run.entryLoad[1];

        std::printf("   %6.3f  %6.2f  %6.2f  %9.3f  %10.4f  %9.2f | %6.0f  %6.0f  %6.0f  %6.0f | %6.0f  %6.0f"
                    "  %6.0f   %7.0f\n",
                    sample.time, sample.speed, sample.longitudinalAcceleration, 1000.0 * sample.pitch,
                    sample.pitchRate, sample.pitchAcceleration, sample.load[0], sample.load[1], sample.load[2],
                    sample.load[3], front, rear, front + rear, front - entryFront);
    }
}

void reportRearSuspension(const Run& run, const double until, const int every, const std::size_t corner)
{
    std::printf("\n    t (s)   travel      q       qdot    shaftComp  shaftVel |  damper  viscous friction  spring"
                "   bump   droop |   tyre  contact\n");
    std::printf("             (mm)     (rad)    (rad/s)     (mm)      (m/s)   |   (N)     (N)     (N)      (N)"
                "     (N)     (N)  |   (N)\n");

    for (auto index = std::size_t{0}; index < run.samples.size(); index++)
    {
        const auto& sample = run.samples[index];
        if (sample.time > until)
        {
            break;
        }
        if (static_cast<int>(index) % every != 0)
        {
            continue;
        }

        std::printf("   %6.3f  %7.2f  %8.5f  %8.4f  %9.3f  %8.5f | %7.1f %7.1f %7.1f %8.1f %6.1f %6.1f | %6.0f  %s\n",
                    sample.time, 1000.0 * sample.wheelTravel[corner], sample.q[corner], sample.qdot[corner],
                    1000.0 * sample.shaftCompression[corner], sample.shaftVelocity[corner], sample.shaftForce[corner],
                    sample.viscous[corner], sample.friction[corner], sample.spring[corner], sample.bumpStop[corner],
                    sample.droopStop[corner], sample.load[corner], sample.inContact[corner] ? "yes" : "OFF");
    }
}

void reportGeneralised(const Run& run, const double until, const int every, const std::size_t corner)
{
    std::printf("\n    t (s)   Q_spring   Q_damper   Q_stops    Q_tyre   Q_unsprung    Q_total    dL/dq    dTr/dq\n");
    std::printf("            (N.m/rad)  (N.m/rad) (N.m/rad) (N.m/rad)  (N.m/rad)   (N.m/rad)  (m/rad)   (m/rad)\n");

    for (auto index = std::size_t{0}; index < run.samples.size(); index++)
    {
        const auto& sample = run.samples[index];
        if (sample.time > until)
        {
            break;
        }
        if (static_cast<int>(index) % every != 0)
        {
            continue;
        }

        std::printf("   %6.3f  %9.2f  %9.2f  %8.2f  %9.2f  %10.2f  %10.2f  %8.5f  %8.5f\n", sample.time,
                    sample.generalisedSpring[corner], sample.generalisedDamper[corner], sample.generalisedStops[corner],
                    sample.generalisedTyre[corner], sample.generalisedUnsprung[corner],
                    sample.generalisedTotal[corner], sample.jacobian[corner], sample.travelPerAngle[corner]);
    }
}

void reportSummary(const char* label, const Run& run)
{
    const auto index = run.minimumRearLoadIndex();

    std::printf("\n  --- %s ---\n", label);
    std::printf("    stopped %s, on plate %s, distance %.4f m, %zu ticks recorded\n", run.stopped ? "yes" : "NO",
                run.onPlate ? "yes" : "NO", run.distance, run.samples.size());
    std::printf("    entry: speed %.6f m/s, height %.6f m, loads %.2f %.2f %.2f %.2f N\n", run.entrySpeed,
                run.entryHeight, run.entryLoad[0], run.entryLoad[1], run.entryLoad[2], run.entryLoad[3]);
    std::printf("    minimum rear tyre load    %9.2f N at t = %.4f s\n", run.minimumRearLoad(),
                run.samples.empty() ? 0.0 : run.samples[index].time);
    std::printf("    peak |pitch|              %9.4f mrad\n", 1000.0 * run.peakPitch());
    std::printf("    peak rear shaft extension %9.3f mm\n", 1000.0 * run.peakRearExtension());
    std::printf("    rear airborne ticks       %9d\n", run.airborneRearTicks());
    if (!run.samples.empty())
    {
        const auto& at = run.samples[index];
        std::printf("    at that instant: rear shaft velocity %.5f m/s, qdot %.4f rad/s, damper %.1f N,\n",
                    at.shaftVelocity[rearLeft], at.qdot[rearLeft], at.shaftForce[rearLeft]);
        std::printf("                     spring %.1f N, droop stop %.1f N, extension %.3f mm, pitch %.3f mrad\n",
                    at.spring[rearLeft], at.droopStop[rearLeft], -1000.0 * at.shaftCompression[rearLeft],
                    1000.0 * at.pitch);
    }
}

} // namespace

TEST_CASE("the braking event that unloads the rear, recorded tick by tick", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    std::printf("\n=== the four-wheels fixture's own manoeuvre: 100 km/h, step to 0.50 pedal, assists off ===\n");
    std::printf("  Settle 1440 ticks, roll 180 ticks at 100 km/h, then a STEP pedal. Nothing about\n");
    std::printf("  the manoeuvre is altered here.\n");

    const auto half = runBraking(built.value(), built.value(), world.value(), 0.50);
    reportSummary("0.50 pedal, assists off (the four-wheels case's scenario)", half);

    std::printf("\n  ##### body and axle loads, first 1.2 s, every 4th tick (90 Hz) #####\n");
    reportBody(half, 1.2, 4);

    std::printf("\n  ##### rear-left suspension, first 1.2 s, every 4th tick #####\n");
    reportRearSuspension(half, 1.2, 4, rearLeft);

    std::printf("\n  ##### rear-left force path, first 1.2 s, every 4th tick #####\n");
    reportGeneralised(half, 1.2, 4, rearLeft);

    const auto full = runBraking(built.value(), built.value(), world.value(), 1.0);
    reportSummary("1.00 pedal, assists off", full);

    std::printf("\n  ##### rear-left suspension at full pedal, first 0.8 s, every 4th tick #####\n");
    reportRearSuspension(full, 0.8, 4, rearLeft);

    std::printf("\n=== what happens immediately before minimum rear load (0.50 pedal) ===\n");
    const auto centre = half.minimumRearLoadIndex();
    const auto from = centre > 60 ? centre - 60 : std::size_t{0};
    const auto to = std::min(half.samples.size(), centre + 20);

    std::printf("\n    t (s)   tyre RL   shaftVel   damper   spring   Q_damper   Q_spring   Q_tyre    pitchRate\n");
    for (auto index = from; index < to; index += 2)
    {
        const auto& sample = half.samples[index];
        std::printf("   %6.4f  %8.1f  %9.5f  %8.1f %8.1f  %9.2f  %9.2f  %8.2f  %9.4f%s\n", sample.time,
                    sample.load[rearLeft], sample.shaftVelocity[rearLeft], sample.shaftForce[rearLeft],
                    sample.spring[rearLeft], sample.generalisedDamper[rearLeft], sample.generalisedSpring[rearLeft],
                    sample.generalisedTyre[rearLeft], sample.pitchRate, index == centre ? "   <== MINIMUM" : "");
    }

    std::printf("\n  --- the accounting AT the minimum, with signs spelled out ---\n");
    {
        const auto& at = half.samples[centre];
        const auto ratio = std::abs(at.jacobian[rearLeft] / std::max(at.travelPerAngle[rearLeft], 1e-12));

        std::printf("    SIGN RULE: a POSITIVE generalised force drives q toward BUMP, which pulls the\n");
        std::printf("    wheel UP toward the body and therefore UNLOADS the tyre. A negative one pushes\n");
        std::printf("    the wheel DOWN onto the road and loads it.\n\n");
        std::printf("      Q_spring     %+10.2f N.m/rad   pushes the wheel DOWN (loads)\n",
                    at.generalisedSpring[rearLeft]);
        std::printf("      Q_damper     %+10.2f N.m/rad   pulls  the wheel UP   (UNLOADS)\n",
                    at.generalisedDamper[rearLeft]);
        std::printf("      Q_stops      %+10.2f N.m/rad   %s\n", at.generalisedStops[rearLeft],
                    at.generalisedStops[rearLeft] == 0.0 ? "the droop stop is NOT touched" : "a stop is engaged");
        std::printf("      Q_unsprung   %+10.2f N.m/rad   the corner's own weight, DOWN\n",
                    at.generalisedUnsprung[rearLeft]);
        std::printf("      Q_tyre       %+10.2f N.m/rad   the road, UP — and it has reached zero\n",
                    at.generalisedTyre[rearLeft]);
        std::printf("      Q_total      %+10.2f N.m/rad\n", at.generalisedTotal[rearLeft]);
        std::printf("\n      the damper offsets %.1f%% of the spring's push-down at this instant\n",
                    100.0 * at.generalisedDamper[rearLeft] / std::max(-at.generalisedSpring[rearLeft], 1e-12));
        std::printf("      damper shaft force %.1f N, which at r = %.4f is %.1f N pulling the wheel up\n",
                    at.shaftForce[rearLeft], ratio, -at.shaftForce[rearLeft] * ratio);
        std::printf("      against a rolling static corner load of %.1f N\n", half.entryLoad[rearLeft]);
        std::printf("      shaft extension here %.3f mm, against a droop stop gap of %.1f mm\n",
                    -1000.0 * at.shaftCompression[rearLeft], 1000.0 * built->corners[rearLeft].droopStop.gap);
    }
}

TEST_CASE("the transient's frequency, measured on four signals rather than quoted", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto run = runBraking(built.value(), built.value(), world.value(), 0.50);
    REQUIRE(run.samples.size() > 500);

    // The window starts one tick after the pedal — a step pedal's first tick is a discontinuity, and
    // including it puts a broadband edge into every periodogram — and ends at 1.2 s, by which time
    // the stop still has a second to run so no end effect from the car halting is in it.
    const auto from = std::size_t{1};
    const auto to = std::size_t{432};

    std::printf("\n=== the braking transient, spectrally ===\n");
    std::printf("  Window %.4f s to %.4f s (%zu samples at 360 Hz), linearly detrended.\n", run.samples[from].time,
                run.samples[to - 1].time, to - from);
    std::printf("  Frequency resolution of the window itself is 1/T = %.3f Hz; the search grid is\n",
                1.0 / (static_cast<double>(to - from) * tick));
    std::printf("  finer than that and does NOT buy resolution, it only locates the peak.\n");

    struct Channel
    {
        const char* name;
        std::size_t what;
        std::size_t corner;
    };

    const auto channels = std::array{Channel{"pitch angle", 0, rearLeft}, Channel{"pitch rate", 1, rearLeft},
                                     Channel{"rear-left wheel travel", 2, rearLeft},
                                     Channel{"rear-left tyre load", 3, rearLeft},
                                     Channel{"rear-left shaft velocity", 4, rearLeft},
                                     Channel{"front-left tyre load", 3, frontLeft}};

    std::printf("\n    signal                      dominant (Hz)   period (ms)   peaks   mean peak gap (ms)"
                "   log decrement\n");

    auto dominant = std::array<double, 6>{};

    for (auto index = std::size_t{0}; index < channels.size(); index++)
    {
        const auto window = detrended(run.channel(channels[index].what, channels[index].corner), from, to);
        const auto frequency = dominantFrequency(window, 0.5, 20.0);
        dominant[index] = frequency;

        const auto peaks = prominentMaxima(window, 40);
        auto gap = 0.0;
        auto decrement = 0.0;
        auto decrementCount = 0;

        if (peaks.size() >= 2)
        {
            gap = tick * static_cast<double>(peaks.back().first - peaks.front().first) /
                  static_cast<double>(peaks.size() - 1);

            for (auto step = std::size_t{0}; step + 1 < peaks.size(); step++)
            {
                if (peaks[step].second > 0.0 && peaks[step + 1].second > 0.0)
                {
                    decrement += std::log(peaks[step].second / peaks[step + 1].second);
                    decrementCount++;
                }
            }
        }

        std::printf("    %-26s  %12.3f   %11.2f   %5zu   %18.2f   %13.4f\n", channels[index].name, frequency,
                    1000.0 / std::max(frequency, 1e-9), peaks.size(), 1000.0 * gap,
                    decrementCount > 0 ? decrement / static_cast<double>(decrementCount) : 0.0);
    }

    std::printf("\n  --- cycle-to-cycle stability, on pitch rate (the cleanest oscillator here) ---\n");
    {
        const auto window = detrended(run.channel(1, rearLeft), from, to);
        const auto peaks = prominentMaxima(window, 40);

        std::printf("    peaks above a quarter of the window's RMS and at least 40 samples apart.\n");
        std::printf("    peak #   index    t (s)      value      gap to next (ms)   implied f (Hz)\n");
        for (auto index = std::size_t{0}; index < peaks.size() && index < 14; index++)
        {
            const auto gap = index + 1 < peaks.size()
                                 ? tick * static_cast<double>(peaks[index + 1].first - peaks[index].first)
                                 : 0.0;
            std::printf("    %6zu  %6zu  %7.4f  %10.5f  %18.2f   %14.3f\n", index, peaks[index].first,
                        run.samples[from + peaks[index].first].time, peaks[index].second, 1000.0 * gap,
                        gap > 0.0 ? 1.0 / gap : 0.0);
        }
    }

    std::printf("\n  --- phase, at the pitch-rate peak frequency ---\n");
    {
        const auto reference = dominant[1];
        const auto pitchWindow = detrended(run.channel(0, rearLeft), from, to);
        const auto travelWindow = detrended(run.channel(2, rearLeft), from, to);
        const auto forceWindow = detrended(run.channel(5, rearLeft), from, to);
        const auto velocityWindow = detrended(run.channel(4, rearLeft), from, to);
        const auto loadWindow = detrended(run.channel(3, rearLeft), from, to);

        const auto pitchPhase = phasorAt(pitchWindow, reference).second;
        const auto travelPhase = phasorAt(travelWindow, reference).second;
        const auto forcePhase = phasorAt(forceWindow, reference).second;
        const auto velocityPhase = phasorAt(velocityWindow, reference).second;
        const auto loadPhase = phasorAt(loadWindow, reference).second;

        const auto wrap = [](const double radians)
        {
            auto value = radians;
            while (value > 3.14159265358979323846)
            {
                value -= 2.0 * 3.14159265358979323846;
            }
            while (value < -3.14159265358979323846)
            {
                value += 2.0 * 3.14159265358979323846;
            }

            return value * 180.0 / 3.14159265358979323846;
        };

        std::printf("    reference frequency                          %8.3f Hz\n", reference);
        std::printf("    rear wheel travel leads pitch angle by       %8.2f deg\n", wrap(travelPhase - pitchPhase));
        std::printf("    rear tyre load leads pitch angle by          %8.2f deg\n", wrap(loadPhase - pitchPhase));
        std::printf("    rear damper force leads shaft velocity by    %8.2f deg\n", wrap(forcePhase - velocityPhase));
        std::printf("    rear tyre load leads rear wheel travel by    %8.2f deg\n", wrap(loadPhase - travelPhase));
        std::printf("\n    The damper/velocity figure is the one that cannot be otherwise: the force is a\n");
        std::printf("    MEMORYLESS function of the instantaneous velocity, so it can only be in phase\n");
        std::printf("    or in antiphase with it, and it has no mechanism to lag. A damper in this model\n");
        std::printf("    cannot amplify an oscillation by phase; it can only change WHERE the wheel is,\n");
        std::printf("    and that is a different mechanism from a phase lag.\n");
    }

    std::printf("\n  --- is the minimum on the first excursion or after oscillation develops? ---\n");
    {
        const auto index = run.minimumRearLoadIndex();
        const auto window = detrended(run.channel(3, rearLeft), from, to);
        const auto peaks = prominentMaxima(window, 40);

        std::printf("    minimum rear load at sample %zu (t = %.4f s)\n", index, run.samples[index].time);
        std::printf("    rear first loses contact at t = %.4f s, for %d ticks, in %d episodes\n",
                    run.timeToFirstLift(), run.firstEpisodeTicks(), run.liftEpisodes());
        std::printf("    first local maximum of the detrended load at sample %zu\n",
                    peaks.empty() ? std::size_t{0} : from + peaks.front().first);

        auto crossings = 0;
        for (auto step = from + 1; step < index; step++)
        {
            if ((run.samples[step].pitchRate > 0.0) != (run.samples[step - 1].pitchRate > 0.0))
            {
                crossings++;
            }
        }
        std::printf("    pitch-rate sign changes before the minimum: %d\n", crossings);
        std::printf("    (0 or 1 means the minimum is on the FIRST excursion; more means the\n");
        std::printf("     oscillation had to develop before the load got there.)\n");
    }

    std::printf("\n  --- the SAME stop, later and quieter: 1.2 s to 2.4 s ---\n");
    std::printf("  The early window contains the wheel leaving the road, which is a hard\n");
    std::printf("  nonlinearity: for the ticks it is airborne the corner is a free mass on a spring\n");
    std::printf("  and the coupled body model does not describe it. This window has no lift in it on\n");
    std::printf("  the shipped car, so a frequency read here is the LINEAR ring-down and is the one\n");
    std::printf("  comparable with the modal table.\n");
    {
        const auto lateFrom = std::size_t{432};
        const auto lateTo = std::min(run.samples.size(), std::size_t{864});

        if (lateTo > lateFrom + 100)
        {
            auto lifts = 0;
            for (auto index = lateFrom; index < lateTo; index++)
            {
                if (!run.samples[index].inContact[rearLeft] || !run.samples[index].inContact[rearRight])
                {
                    lifts++;
                }
            }

            std::printf("    window %.4f s to %.4f s, %d airborne rear ticks in it\n", run.samples[lateFrom].time,
                        run.samples[lateTo - 1].time, lifts);
            std::printf("    signal                      dominant (Hz)   period (ms)   prominent peaks   log dec\n");

            for (auto index = std::size_t{0}; index < channels.size(); index++)
            {
                const auto window = detrended(run.channel(channels[index].what, channels[index].corner), lateFrom, lateTo);
                const auto peaks = prominentMaxima(window, 40);

                auto decrement = 0.0;
                auto count = 0;
                for (auto step = std::size_t{0}; step + 1 < peaks.size(); step++)
                {
                    if (peaks[step].second > 0.0 && peaks[step + 1].second > 0.0)
                    {
                        decrement += std::log(peaks[step].second / peaks[step + 1].second);
                        count++;
                    }
                }

                const auto frequency = dominantFrequency(window, 0.5, 20.0);
                std::printf("    %-26s  %12.3f   %11.2f   %15zu   %7.4f\n", channels[index].name, frequency,
                            1000.0 / std::max(frequency, 1e-9), peaks.size(),
                            count > 0 ? decrement / static_cast<double>(count) : 0.0);
            }
        }
    }
}

TEST_CASE("the small-signal modes the production parameters already imply", "[.rear-damper]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    REQUIRE(built->sprung.size() == 1);

    const auto& sprung = built->sprung.front();
    const auto sprungMass = sprung.mass;
    const auto pitchInertia = sprung.inertia[0][0];

    std::printf("\n=== linearised modes, from the shipped numbers and nothing fitted ===\n");
    std::printf("  SIMPLIFICATIONS, stated: rigid axles reduced to two independent corner pairs; the\n");
    std::printf("  linkage taken at its DESIGN motion ratio and treated as constant; anti-roll bars\n");
    std::printf("  ignored (they carry no load in a symmetric pitch); tyre damping ignored; no\n");
    std::printf("  aerodynamic, driveline or brake coupling; the damper linearised at its SLOW slope,\n");
    std::printf("  which is the largest slope it has. Every number below is small-signal and none of\n");
    std::printf("  them was adjusted to match anything measured.\n");

    auto wheelRate = std::array<double, 2>{};
    auto rideRate = std::array<double, 2>{};
    auto reboundShaftSlope = std::array<double, 2>{};
    auto bumpShaftSlope = std::array<double, 2>{};
    auto damperRatio = std::array<double, 2>{};
    auto arm = std::array<double, 2>{};

    for (auto axle = std::size_t{0}; axle < 2; axle++)
    {
        const auto& corner = built->corners[axles[axle].index];
        const auto kinematics = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0, 0.0);
        REQUIRE(kinematics.has_value());

        const auto ratio = std::abs(kinematics->motionRatio);
        damperRatio[axle] = ratio;
        wheelRate[axle] = corner.springRate * ratio * ratio;
        rideRate[axle] = wheelRate[axle] * corner.tireVerticalRate / (wheelRate[axle] + corner.tireVerticalRate);
        reboundShaftSlope[axle] = curveSlope(corner.damper, -0.02);
        bumpShaftSlope[axle] = curveSlope(corner.damper, 0.02);
        arm[axle] = corner.hardpoints.wheelCentre.z - sprung.centre.z;

        std::printf("\n  --- %s ---\n", axles[axle].name);
        std::printf("    shaft spring rate                  %10.1f N/m\n", corner.springRate);
        std::printf("    damper motion ratio                %10.6f\n", ratio);
        std::printf("    wheel rate                         %10.1f N/m\n", wheelRate[axle]);
        std::printf("    tyre vertical rate                 %10.1f N/m\n", corner.tireVerticalRate);
        std::printf("    ride rate (spring in series tyre)  %10.1f N/m\n", rideRate[axle]);
        std::printf("    unsprung mass                      %10.2f kg\n", corner.unsprungMass);
        std::printf("    slow rebound slope at the shaft    %10.1f N.s/m\n", reboundShaftSlope[axle]);
        std::printf("    slow bump slope at the shaft       %10.1f N.s/m\n", bumpShaftSlope[axle]);
        std::printf("    wheel-referred rebound coefficient %10.1f N.s/m\n", reboundShaftSlope[axle] * ratio * ratio);
        std::printf("    wheel-referred bump coefficient    %10.1f N.s/m\n", bumpShaftSlope[axle] * ratio * ratio);
        std::printf("    axle station relative to sprung CG %10.4f m (+ forward)\n", arm[axle]);
    }

    // Sprung mass on each axle, by statics about the other one.
    const auto a = arm[0];
    const auto b = -arm[1];
    const auto wheelbase = a + b;
    const auto sprungFront = sprungMass * b / wheelbase;
    const auto sprungRear = sprungMass - sprungFront;

    std::printf("\n  --- the sprung body ---\n");
    std::printf("    sprung mass                        %10.2f kg\n", sprungMass);
    std::printf("    pitch inertia about the sprung CG  %10.2f kg.m^2\n", pitchInertia);
    std::printf("    sprung CG height / station         %10.4f / %.4f m\n", sprung.centre.y, sprung.centre.z);
    std::printf("    front arm a                        %10.4f m\n", a);
    std::printf("    rear  arm b                        %10.4f m\n", b);
    std::printf("    sprung on the front axle           %10.2f kg (%.2f per corner)\n", sprungFront,
                0.5 * sprungFront);
    std::printf("    sprung on the rear axle            %10.2f kg (%.2f per corner)\n", sprungRear, 0.5 * sprungRear);
    std::printf("    dynamic index k^2/(ab)             %10.4f\n", pitchInertia / (sprungMass * a * b));

    const auto twoPi = 6.283185307179586;

    std::printf("\n  --- single-axle ride modes (each axle alone, on its own sprung share) ---\n");
    std::printf("    axle    on WHEEL rate (Hz)   on RIDE rate (Hz)   damper zeta (rebound)   zeta (bump)\n");
    for (auto axle = std::size_t{0}; axle < 2; axle++)
    {
        const auto mass = 0.5 * (axle == 0 ? sprungFront : sprungRear);
        const auto onWheel = std::sqrt(wheelRate[axle] / mass) / twoPi;
        const auto onRide = std::sqrt(rideRate[axle] / mass) / twoPi;
        const auto wheelRebound = reboundShaftSlope[axle] * damperRatio[axle] * damperRatio[axle];
        const auto wheelBump = bumpShaftSlope[axle] * damperRatio[axle] * damperRatio[axle];

        std::printf("    %-6s  %17.4f   %17.4f   %21.4f   %11.4f\n", axles[axle].name, onWheel, onRide,
                    wheelRebound / (2.0 * std::sqrt(rideRate[axle] * mass)),
                    wheelBump / (2.0 * std::sqrt(rideRate[axle] * mass)));
    }

    std::printf("\n  --- the coupled heave/pitch pair, on ride rates ---\n");
    {
        const auto kf = 2.0 * rideRate[0];
        const auto kr = 2.0 * rideRate[1];
        const auto k11 = kf + kr;
        const auto k12 = kf * a - kr * b;
        const auto k22 = kf * a * a + kr * b * b;

        const auto a11 = k11 / sprungMass;
        const auto a12 = k12 / sprungMass;
        const auto a21 = k12 / pitchInertia;
        const auto a22 = k22 / pitchInertia;

        const auto trace = a11 + a22;
        const auto determinant = a11 * a22 - a12 * a21;
        const auto discriminant = std::max(0.0, trace * trace - 4.0 * determinant);
        const auto root = std::sqrt(discriminant);
        const auto lambdaHigh = 0.5 * (trace + root);
        const auto lambdaLow = 0.5 * (trace - root);

        std::printf("    K = [ %10.1f  %11.1f ; %11.1f  %11.1f ]  (N/m, N/rad, N.m/rad)\n", k11, k12, k12, k22);
        std::printf("    coupled mode 1  %8.4f Hz   (omega^2 = %.2f)\n", std::sqrt(lambdaLow) / twoPi, lambdaLow);
        std::printf("    coupled mode 2  %8.4f Hz   (omega^2 = %.2f)\n", std::sqrt(lambdaHigh) / twoPi, lambdaHigh);
        std::printf("    mode shapes (heave, pitch): low  (%.4f, %.4f)   high (%.4f, %.4f)\n", 1.0,
                    (lambdaLow - a11) / std::max(std::abs(a12), 1e-12) * (a12 < 0.0 ? -1.0 : 1.0), 1.0,
                    (lambdaHigh - a11) / std::max(std::abs(a12), 1e-12) * (a12 < 0.0 ? -1.0 : 1.0));
        std::printf("    on WHEEL rates instead: ");
        {
            const auto kfw = 2.0 * wheelRate[0];
            const auto krw = 2.0 * wheelRate[1];
            const auto w11 = (kfw + krw) / sprungMass;
            const auto w12 = (kfw * a - krw * b) / sprungMass;
            const auto w21 = (kfw * a - krw * b) / pitchInertia;
            const auto w22 = (kfw * a * a + krw * b * b) / pitchInertia;
            const auto wt = w11 + w22;
            const auto wd = w11 * w22 - w12 * w21;
            const auto wr = std::sqrt(std::max(0.0, wt * wt - 4.0 * wd));
            std::printf("%.4f Hz and %.4f Hz\n", std::sqrt(0.5 * (wt - wr)) / twoPi, std::sqrt(0.5 * (wt + wr)) / twoPi);
        }
    }

    std::printf("\n  --- unsprung / wheel-hop modes ---\n");
    std::printf("    axle    hop (Hz)   corner-DOF (Hz)   corner zeta (rebound)   generalised inertia\n");
    for (auto axle = std::size_t{0}; axle < 2; axle++)
    {
        const auto& corner = built->corners[axles[axle].index];
        const auto solved = raceengine::solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        REQUIRE(solved.has_value());

        const auto kinematics = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0, 0.0);
        REQUIRE(kinematics.has_value());

        const auto hop = std::sqrt((wheelRate[axle] + corner.tireVerticalRate) / corner.unsprungMass) / twoPi;

        // The model's OWN corner degree of freedom: generalised inertia is the unsprung mass through
        // the wheel centre's whole Jacobian (`geometricLoadPath` is on for this car), and the
        // generalised stiffness is the shaft spring and the tyre each through their own Jacobian.
        const auto generalisedInertia =
            corner.unsprungMass * glm::dot(solved->wheelCentrePerAngle, solved->wheelCentrePerAngle);
        const auto generalisedStiffness =
            corner.springRate * kinematics->lengthPerAngle * kinematics->lengthPerAngle +
            corner.tireVerticalRate * solved->travelPerAngle * solved->travelPerAngle;
        const auto generalisedDamping =
            reboundShaftSlope[axle] * kinematics->lengthPerAngle * kinematics->lengthPerAngle;

        std::printf("    %-6s  %8.3f   %15.3f   %21.4f   %14.6f kg.m^2\n", axles[axle].name, hop,
                    std::sqrt(generalisedStiffness / generalisedInertia) / twoPi,
                    generalisedDamping / (2.0 * std::sqrt(generalisedStiffness * generalisedInertia)),
                    generalisedInertia);
    }

    std::printf("\n  The question this table answers is only whether a measured frequency corresponds\n");
    std::printf("  to a mode the model already has. Nothing above was adjusted to make it do so.\n");
}

TEST_CASE("where on its own curve the rear damper actually works during the stop", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto& rear = built->corners[rearLeft];
    const auto kinematics = solveDamperKinematics(rear.hardpoints, damperElementOf(rear.hardpoints), 0.0, 0.0);
    REQUIRE(kinematics.has_value());

    const auto ratio = std::abs(kinematics->motionRatio);
    const auto jacobian = kinematics->lengthPerAngle;
    const auto reboundKnee = -rear.damper.points[1].x;
    const auto bumpKnee = rear.damper.points[3].x;

    std::printf("\n=== the rear damper's operating region, 0.50 and 1.00 pedal ===\n");
    std::printf("  Rebound knee at %.6f m/s of SHAFT speed, bump knee at %.6f m/s.\n", reboundKnee, bumpKnee);

    for (const auto pedal : {0.50, 1.0})
    {
        const auto run = runBraking(built.value(), built.value(), world.value(), pedal);
        const auto index = run.minimumRearLoadIndex();

        auto lowest = 0.0;
        auto highest = 0.0;
        auto reboundSamples = 0;
        auto reboundPastKnee = 0;
        auto bumpSamples = 0;
        auto bumpPastKnee = 0;

        for (const auto& sample : run.samples)
        {
            for (const auto corner : {rearLeft, rearRight})
            {
                const auto velocity = sample.shaftVelocity[corner];
                lowest = std::min(lowest, velocity);
                highest = std::max(highest, velocity);

                if (velocity < 0.0)
                {
                    reboundSamples++;
                    if (-velocity >= reboundKnee)
                    {
                        reboundPastKnee++;
                    }
                }
                else if (velocity > 0.0)
                {
                    bumpSamples++;
                    if (velocity >= bumpKnee)
                    {
                        bumpPastKnee++;
                    }
                }
            }
        }

        const auto total = reboundSamples + bumpSamples;

        std::printf("\n  --- %.2f pedal ---\n", pedal);
        std::printf("    samples (both rear corners)                 %8d\n", total);
        std::printf("    minimum shaft velocity (deepest REBOUND)    %8.5f m/s  = %.3f of the rebound knee\n", lowest,
                    -lowest / reboundKnee);
        std::printf("    maximum shaft velocity (deepest BUMP)       %8.5f m/s  = %.3f of the bump knee\n", highest,
                    highest / bumpKnee);
        std::printf("    velocity at minimum rear tyre load          %8.5f m/s  = %.3f of the rebound knee\n",
                    run.samples[index].shaftVelocity[rearLeft], -run.samples[index].shaftVelocity[rearLeft] / reboundKnee);
        std::printf("    rebound samples  %6d  (%.2f%% of all), of which past the knee %6d (%.2f%% of rebound)\n",
                    reboundSamples, 100.0 * static_cast<double>(reboundSamples) / static_cast<double>(std::max(total, 1)),
                    reboundPastKnee,
                    100.0 * static_cast<double>(reboundPastKnee) / static_cast<double>(std::max(reboundSamples, 1)));
        std::printf("    bump    samples  %6d  (%.2f%% of all), of which past the knee %6d (%.2f%% of bump)\n",
                    bumpSamples, 100.0 * static_cast<double>(bumpSamples) / static_cast<double>(std::max(total, 1)),
                    bumpPastKnee,
                    100.0 * static_cast<double>(bumpPastKnee) / static_cast<double>(std::max(bumpSamples, 1)));

        // **The split that decides whether the knee is a cause or a consequence.** After lift-off the
        // corner is a free mass on a spring, so its shaft speed is a reading of the wheel falling,
        // not of the event that unloaded it.
        {
            const auto lift = run.firstLiftIndex();
            auto preLiftPeak = 0.0;
            auto preLiftSamples = 0;
            auto preLiftPastKnee = 0;

            for (auto step = std::size_t{0}; step < lift && step < run.samples.size(); step++)
            {
                for (const auto corner : {rearLeft, rearRight})
                {
                    const auto velocity = run.samples[step].shaftVelocity[corner];
                    if (velocity < 0.0)
                    {
                        preLiftSamples++;
                        preLiftPeak = std::max(preLiftPeak, -velocity);
                        if (-velocity >= reboundKnee)
                        {
                            preLiftPastKnee++;
                        }
                    }
                }
            }

            std::printf("\n    rear first loses contact at t = %.4f s (sample %zu of %zu)\n", run.timeToFirstLift(),
                        lift, run.samples.size());
            std::printf("    BEFORE that instant: deepest rebound %.5f m/s = %.3f of the knee\n", preLiftPeak,
                        preLiftPeak / reboundKnee);
            std::printf("    BEFORE that instant: %d rebound samples, %d past the knee (%.2f%%)\n", preLiftSamples,
                        preLiftPastKnee,
                        100.0 * static_cast<double>(preLiftPastKnee) / static_cast<double>(std::max(preLiftSamples, 1)));
        }

        if (pedal == 0.50)
        {
            std::printf("\n    the trajectory through the curve, every 6th tick to 1.0 s (rear left):\n");
            std::printf("      t (s)   shaftVel   branch      F (N)   dF/dv (N.s/m)   c_wheel    c_q\n");
            std::printf("              (m/s)                          shaft            (N.s/m)  (N.m.s/rad)\n");
            for (auto step = std::size_t{0}; step < run.samples.size() && run.samples[step].time <= 1.0; step += 6)
            {
                const auto velocity = run.samples[step].shaftVelocity[rearLeft];
                const auto slope = curveSlope(rear.damper, velocity);
                const char* branch = velocity < -reboundKnee   ? "reb-FAST"
                                     : velocity < 0.0          ? "reb-slow"
                                     : velocity < bumpKnee     ? "bmp-slow"
                                                               : "bmp-FAST";

                std::printf("     %6.3f  %9.5f   %-9s %8.1f  %13.1f  %9.1f  %9.2f\n", run.samples[step].time, velocity,
                            branch, run.samples[step].viscous[rearLeft], slope, slope * ratio * ratio,
                            slope * jacobian * jacobian);
            }
        }
    }

    std::printf("\n  --- the curve at representative velocities, for reference ---\n");
    std::printf("      v (m/s)      F (N)    dF/dv shaft   c_wheel (N.s/m)   c_q (N.m.s/rad)\n");
    for (const auto velocity : {-0.400, -0.300, -0.200, -0.1382, -0.100, -0.050, -0.020, -0.005, 0.005, 0.020, 0.050,
                                0.0987, 0.200, 0.400})
    {
        const auto slope = curveSlope(rear.damper, velocity);
        std::printf("     %8.4f  %9.1f  %13.1f  %16.1f  %16.2f\n", velocity, rear.damper.at(velocity), slope,
                    slope * ratio * ratio, slope * jacobian * jacobian);
    }
}

namespace
{

struct Metrics
{
    double minimumRearLoad = 0.0;
    double minimumRearTime = 0.0;
    double peakPitch = 0.0;
    double pitchFrequency = 0.0;
    double pitchDecrement = 0.0;
    double peakRearExtension = 0.0;
    double peakRearQdot = 0.0;
    double peakRearShaftVelocity = 0.0;
    int airborneRear = 0;
    double distance = 0.0;
    bool stopped = false;
    double timeToFirstLift = 0.0;
    int firstEpisodeTicks = 0;
    int liftEpisodes = 0;
    double rearLoadDeficit = 0.0;
    // The deepest rebound the shaft reaches BEFORE the tyre ever leaves the road. Past lift-off the
    // wheel is a free mass on a spring and its shaft speed says nothing about what caused the lift.
    double preLiftPeakRebound = 0.0;
};

[[nodiscard]] Metrics measure(const Run& run)
{
    auto metrics = Metrics{};
    metrics.minimumRearLoad = run.minimumRearLoad();
    metrics.timeToFirstLift = run.timeToFirstLift();
    metrics.firstEpisodeTicks = run.firstEpisodeTicks();
    metrics.liftEpisodes = run.liftEpisodes();
    metrics.rearLoadDeficit = run.rearLoadDeficit();

    {
        const auto lift = run.firstLiftIndex();
        for (auto index = std::size_t{0}; index < lift && index < run.samples.size(); index++)
        {
            for (const auto corner : {rearLeft, rearRight})
            {
                metrics.preLiftPeakRebound =
                    std::max(metrics.preLiftPeakRebound, -run.samples[index].shaftVelocity[corner]);
            }
        }
    }
    metrics.minimumRearTime = run.samples.empty() ? 0.0 : run.samples[run.minimumRearLoadIndex()].time;
    metrics.peakPitch = run.peakPitch();
    metrics.peakRearExtension = run.peakRearExtension();
    metrics.airborneRear = run.airborneRearTicks();
    metrics.distance = run.distance;
    metrics.stopped = run.stopped;

    for (const auto& sample : run.samples)
    {
        for (const auto corner : {rearLeft, rearRight})
        {
            metrics.peakRearQdot = std::max(metrics.peakRearQdot, std::abs(sample.qdot[corner]));
            metrics.peakRearShaftVelocity = std::max(metrics.peakRearShaftVelocity, std::abs(sample.shaftVelocity[corner]));
        }
    }

    const auto to = std::min(run.samples.size(), std::size_t{432});
    if (to > 40)
    {
        const auto window = detrended(run.channel(1, rearLeft), 1, to);
        metrics.pitchFrequency = dominantFrequency(window, 0.5, 20.0);

        const auto peaks = prominentMaxima(window, 40);
        auto sum = 0.0;
        auto count = 0;
        for (auto index = std::size_t{0}; index + 1 < peaks.size(); index++)
        {
            if (peaks[index].second > 0.0 && peaks[index + 1].second > 0.0)
            {
                sum += std::log(peaks[index].second / peaks[index + 1].second);
                count++;
            }
        }
        metrics.pitchDecrement = count > 0 ? sum / static_cast<double>(count) : 0.0;
    }

    return metrics;
}

void reportHeader()
{
    std::printf("\n    arm                          minRear  liftAt  1stEp  eps  deficit  peakPitch  f_pitch  logDec"
                "  peakExt  preLiftReb  airborne  distance\n");
    std::printf("                                   (N)      (s)   (ticks)       (N.s)     (mrad)     (Hz)"
                "            (mm)     (m/s)     (ticks)    (m)\n");
}

void reportMetrics(const char* label, const Metrics& metrics)
{
    std::printf("    %-28s %7.1f  %6.4f  %5d  %3d  %7.1f  %9.4f  %7.3f  %6.3f  %7.3f  %9.5f  %7d  %8.4f%s\n", label,
                metrics.minimumRearLoad, metrics.timeToFirstLift, metrics.firstEpisodeTicks, metrics.liftEpisodes,
                metrics.rearLoadDeficit, 1000.0 * metrics.peakPitch, metrics.pitchFrequency, metrics.pitchDecrement,
                1000.0 * metrics.peakRearExtension, metrics.preLiftPeakRebound, metrics.airborneRear,
                metrics.distance, metrics.stopped ? "" : "  (NOT STOPPED)");
}

} // namespace

TEST_CASE("DIAGNOSTIC A/B: the rear damper's ordinary damping removed outright", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    std::printf("\n=== A/B: the whole causal size of ordinary rear damping ===\n");
    std::printf("  Arm B keeps the spring, the tyre, both stops and the 25 N of seal friction, and\n");
    std::printf("  replaces the rear viscous curve with a flat zero. It is NOT a candidate setup and\n");
    std::printf("  nothing is installed anywhere: production is arm A in both pedal positions.\n");

    const auto dead = withRearDamper(built.value(), deadCurve());

    for (const auto pedal : {0.50, 1.0})
    {
        const auto shipped = runBraking(built.value(), built.value(), world.value(), pedal);
        const auto removed = runBraking(built.value(), dead, world.value(), pedal);

        std::printf("\n  ##### %.2f pedal, assists off #####\n", pedal);
        reportHeader();
        reportMetrics("A: shipped", measure(shipped));
        reportMetrics("B: rear viscous removed", measure(removed));

        // The two arms must have started from the same place, or nothing above is a comparison.
        CHECK(std::abs(shipped.entrySpeed - removed.entrySpeed) < 1e-12);
        CHECK(std::abs(shipped.entryHeight - removed.entryHeight) < 1e-12);
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            CHECK(std::abs(shipped.entryLoad[index] - removed.entryLoad[index]) < 1e-9);
        }
    }

    std::printf("\n  --- the control: arm the change at tick zero of the SETTLE instead ---\n");
    std::printf("  If the answer moves, the shared settle is doing work and the A/B above is not a\n");
    std::printf("  clean one-variable comparison. If it does not, the settle is inert as claimed.\n");
    {
        const auto lateArm = runBraking(built.value(), dead, world.value(), 0.50);
        const auto earlyArm = runBraking(built.value(), dead, world.value(), 0.50, true);

        reportHeader();
        reportMetrics("B: armed at the pedal", measure(lateArm));
        reportMetrics("B: armed at the settle", measure(earlyArm));
        std::printf("    entry loads, armed at the pedal   %.4f %.4f %.4f %.4f N\n", lateArm.entryLoad[0],
                    lateArm.entryLoad[1], lateArm.entryLoad[2], lateArm.entryLoad[3]);
        std::printf("    entry loads, armed at the settle  %.4f %.4f %.4f %.4f N\n", earlyArm.entryLoad[0],
                    earlyArm.entryLoad[1], earlyArm.entryLoad[2], earlyArm.entryLoad[3]);
    }
}

TEST_CASE("DIAGNOSTIC A/B: rebound against compression, separated", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto& shippedCurve = built->corners[rearLeft].damper;
    const auto bumpOnly = withRearDamper(built.value(), halfCurve(shippedCurve, true, false));
    const auto reboundOnly = withRearDamper(built.value(), halfCurve(shippedCurve, false, true));
    const auto dead = withRearDamper(built.value(), deadCurve());

    std::printf("\n=== A/B: which half of the rear curve owns the behaviour ===\n");
    std::printf("  Seal friction is untouched in every arm. Zeroing a half leaves the surviving\n");
    std::printf("  half's knots bit-identical to production.\n");

    for (const auto pedal : {0.50, 1.0})
    {
        std::printf("\n  ##### %.2f pedal, assists off #####\n", pedal);
        reportHeader();
        reportMetrics("A: shipped", measure(runBraking(built.value(), built.value(), world.value(), pedal)));
        reportMetrics("B: rebound removed, bump kept", measure(runBraking(built.value(), bumpOnly, world.value(), pedal)));
        reportMetrics("C: bump removed, rebound kept", measure(runBraking(built.value(), reboundOnly, world.value(), pedal)));
        reportMetrics("D: both removed", measure(runBraking(built.value(), dead, world.value(), pedal)));
    }
}

TEST_CASE("DIAGNOSTIC: a coarse causal slope on the rebound force", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    std::printf("\n=== rebound sensitivity: 0.5x, 1.0x, 1.5x, and nothing in between ===\n");
    std::printf("  Three points to estimate a slope with. **No optimum is searched for, none is\n");
    std::printf("  reported, and none of these is a proposal.** The knee SPEED does not move; only\n");
    std::printf("  the force at and past it is scaled.\n");

    const auto& shippedCurve = built->corners[rearLeft].damper;

    for (const auto pedal : {0.50, 1.0})
    {
        std::printf("\n  ##### %.2f pedal, assists off #####\n", pedal);
        reportHeader();

        for (const auto multiplier : {0.5, 1.0, 1.5})
        {
            const auto arm = withRearDamper(built.value(), scaledRebound(shippedCurve, multiplier));
            const auto label = std::string("rebound x") + std::to_string(multiplier).substr(0, 3);
            reportMetrics(label.c_str(), measure(runBraking(built.value(), arm, world.value(), pedal)));
        }
    }
}

TEST_CASE("damping uncertainty against motion-ratio uncertainty, separated", "[.rear-damper]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    std::printf("\n=== the two uncertainties, and why they are not the same size ===\n");
    std::printf("  A. the SOURCE CURVE: AC's six numbers per axle, and whether they are wheel-referred.\n");
    std::printf("  B. the GEOMETRY: this car's rear damper mounts are PLACED, not sourced. The design\n");
    std::printf("     ratio is %.6f; VW SSP 515 p.9 puts the real damper on the spring link, which\n",
                std::abs(solveDamperKinematics(built->corners[rearLeft].hardpoints,
                                               damperElementOf(built->corners[rearLeft].hardpoints), 0.0, 0.0)
                             ->motionRatio));
    std::printf("     makes the true ratio strictly below one. Nothing publishes a number. The old\n");
    std::printf("     0.64 is a SPRING figure and is not usable here; 0.78 has no source and is not\n");
    std::printf("     used below.\n");

    std::printf("\n  --- ANALYTIC: what a motion ratio r does to each stage ---\n");
    std::printf("    shaft velocity            v_s = r * v_wheel                     one power of r\n");
    std::printf("    shaft force from curve    F_s = (c_wheel / r^2) * v_s           built by kneedDamper\n");
    std::printf("    force at the wheel        F_w = r * F_s = c_wheel * v_wheel     r CANCELS EXACTLY\n");
    std::printf("    knee, referred to wheel   v_k / r = knee                        r CANCELS EXACTLY\n");
    std::printf("    generalised damping       c_q = (dF/dv) * (dL/dq)^2             r enters twice and\n");
    std::printf("                              and dL/dq = r * (dTravel/dq)          cancels against the\n");
    std::printf("                              so c_q = c_wheel * (dTravel/dq)^2     conversion again\n");
    std::printf("\n    **So under the wheel-referred reading the ordinary damper is EXACTLY invariant\n");
    std::printf("    to r — force, knee and generalised coefficient all.** The placed rear geometry\n");
    std::printf("    cannot be wrong about the damping in a way that matters, because the conversion\n");
    std::printf("    and the linkage divide and multiply by the same two powers of r.\n");
    std::printf("    What r DOES move, and these are separate questions: seal friction (stated at the\n");
    std::printf("    shaft, so ONE power of r reaches the wheel), the stops' gaps (stated at the\n");
    std::printf("    shaft, so wheel travel to engagement is gap/r) and the damper's own stroke.\n");

    std::printf("\n  --- DEMONSTRATION: the curve built at five ratios, referred back to the wheel ---\n");
    std::printf("    r        shaft slow reb   shaft knee   |   wheel slow reb   wheel knee   wheel F at knee\n");
    std::printf("             (N.s/m)          (m/s)        |   (N.s/m)          (m/s)        (N)\n");
    for (const auto ratio : {0.80, 0.85, 0.90, 0.95, 0.9871})
    {
        const auto kinematics = raceengine::DamperKinematics{.length = 0.4, .lengthPerAngle = -0.4, .motionRatio = -ratio};
        const auto curve = kneedDamper(acBumpRate[1], acFastBumpRate[1], acBumpKnee[1], acReboundRate[1],
                                       acFastReboundRate[1], acReboundKnee[1], kinematics);

        const auto shaftKnee = -curve.points[1].x;
        const auto shaftForce = -curve.points[1].y;
        const auto shaftSlope = shaftForce / shaftKnee;

        std::printf("    %6.4f  %14.2f  %11.6f   |  %14.2f  %11.6f  %14.2f\n", ratio, shaftSlope, shaftKnee,
                    shaftSlope * ratio * ratio, shaftKnee / ratio, shaftForce * ratio);
    }

    std::printf("\n  --- what the CONVENTION question is worth, per axle ---\n");
    std::printf("    If AC's numbers were shaft-referred and this model divided by r^2 anyway, the\n");
    std::printf("    car would carry 1/r^2 times the damping it should:\n");
    for (auto axle = std::size_t{0}; axle < 2; axle++)
    {
        const auto& corner = built->corners[axles[axle].index];
        const auto kinematics = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0, 0.0);
        REQUIRE(kinematics.has_value());

        const auto ratio = std::abs(kinematics->motionRatio);
        std::printf("      %-6s r = %.6f  ->  1/r^2 = %.4f, so a convention error is worth %+.2f%%\n",
                    axles[axle].name, ratio, 1.0 / (ratio * ratio), 100.0 * (1.0 / (ratio * ratio) - 1.0));
    }
    std::printf("\n    The rear figure is inside the coarse sensitivity band swept above, so the\n");
    std::printf("    convention ambiguity cannot on its own change a causal verdict at the rear. The\n");
    std::printf("    front figure is an order of magnitude larger and is NOT audited here.\n");
    std::printf("\n    A genuine test of B needs the hardpoints MOVED, which changes ride height, the\n");
    std::printf("    spring rest length, the stops' wheel travel and the corner Jacobian together.\n");
    std::printf("    That is a different experiment and this probe deliberately does not run it.\n");
}

TEST_CASE("the front damper, as a confound rather than as an audit", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto run = runBraking(built.value(), built.value(), world.value(), 0.50);
    const auto index = run.minimumRearLoadIndex();

    std::printf("\n=== front against rear generalised damper force, through the transient ===\n");
    std::printf("  Both are N.m/rad in their own corner's coordinate, so they are NOT directly\n");
    std::printf("  comparable as numbers. The comparable quantity is the force each puts into the\n");
    std::printf("  BODY, which for a corner is the axis force times the motion ratio — printed as\n");
    std::printf("  `wheel-referred damper force` below, in newtons at the contact patch.\n");

    std::printf("\n    t (s)   FL v_shaft   FL F_damp  FL F_wheel |  RL v_shaft   RL F_damp  RL F_wheel |  pitchRate\n");
    std::printf("             (m/s)         (N)         (N)      |   (m/s)        (N)         (N)      |  (rad/s)\n");

    for (auto step = std::size_t{0}; step < run.samples.size() && run.samples[step].time <= 1.0; step += 6)
    {
        const auto& sample = run.samples[step];
        const auto frontRatio = std::abs(sample.jacobian[frontLeft] / std::max(sample.travelPerAngle[frontLeft], 1e-12));
        const auto rearRatio = std::abs(sample.jacobian[rearLeft] / std::max(sample.travelPerAngle[rearLeft], 1e-12));

        std::printf("   %6.3f  %10.5f  %10.1f  %10.1f | %10.5f  %10.1f  %10.1f | %9.4f%s\n", sample.time,
                    sample.shaftVelocity[frontLeft], sample.shaftForce[frontLeft],
                    sample.shaftForce[frontLeft] * frontRatio, sample.shaftVelocity[rearLeft],
                    sample.shaftForce[rearLeft], sample.shaftForce[rearLeft] * rearRatio, sample.pitchRate,
                    step == index ? "  <== min rear load" : "");
    }

    std::printf("\n  --- A/B: the FRONT viscous curve removed, rear untouched ---\n");
    std::printf("  Diagnostic only, to size the front's share of the pitch event. This is not a front\n");
    std::printf("  damper audit and nothing about its source is examined here.\n");

    auto frontDead = built.value();
    frontDead.corners[0].damper = deadCurve();
    frontDead.corners[1].damper = deadCurve();

    auto bothDead = frontDead;
    bothDead.corners[rearLeft].damper = deadCurve();
    bothDead.corners[rearRight].damper = deadCurve();

    reportHeader();
    reportMetrics("A: shipped", measure(run));
    reportMetrics("E: front viscous removed", measure(runBraking(built.value(), frontDead, world.value(), 0.50)));
    reportMetrics("F: front and rear removed", measure(runBraking(built.value(), bothDead, world.value(), 0.50)));
}

TEST_CASE("rear damper invariants", "[.rear-damper]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto& rear = built->corners[rearLeft];
    const auto& front = built->corners[frontLeft];

    // Both rear corners state the same damper. A car whose two rear dampers differ is a defect, not
    // a setup, because nothing in this project's data states an asymmetric one.
    CHECK(built->corners[rearLeft].damper.points == built->corners[rearRight].damper.points);

    // Five knots, ascending in x, with the origin stated. `Curve::at` relies on ascending x and the
    // zero-velocity force being exactly zero relies on (0,0) being a knot.
    REQUIRE(rear.damper.points.size() == 5);
    for (auto index = std::size_t{1}; index < rear.damper.points.size(); index++)
    {
        CHECK(rear.damper.points[index].x > rear.damper.points[index - 1].x);
    }
    CHECK(rear.damper.at(0.0) == 0.0);

    // The curve is monotonic, which is what makes `curveSlopeAt` non-negative everywhere and the
    // implicit integration's `max(0, slope)` clamp inert. A damper with a falling branch would make
    // that clamp load-bearing and nothing here would say so.
    for (auto index = std::size_t{1}; index < rear.damper.points.size(); index++)
    {
        CHECK(rear.damper.points[index].y > rear.damper.points[index - 1].y);
    }

    // Rebound is the stiffer half at low speed on both axles, which is what a road damper does and
    // what AC's own numbers state. If this ever inverts, the transcription has been edited.
    CHECK(-rear.damper.points[1].y / -rear.damper.points[1].x > rear.damper.points[3].y / rear.damper.points[3].x);
    CHECK(-front.damper.points[1].y / -front.damper.points[1].x > front.damper.points[3].y / front.damper.points[3].x);

    // Ends are HELD, not extrapolated: past the last knot the force is constant and the slope is
    // zero. Three tests here because all three are relied on somewhere — the force pass, the
    // implicit damping's slope, and every table in this file.
    CHECK(rear.damper.at(9.0) == rear.damper.points.back().y);
    CHECK(rear.damper.at(-9.0) == rear.damper.points.front().y);
    CHECK(curveSlope(rear.damper, 7.0) == 0.0);

    // The damper is exactly antisymmetric in SIGN but not in magnitude, and the asymmetry is the
    // data's. Stated so that a future symmetric damper cannot arrive unnoticed.
    CHECK(std::abs(rear.damper.points[1].y) != std::abs(rear.damper.points[3].y));

    // The wheel-referred round trip, which is the whole content of the conversion and the thing the
    // motion-ratio argument rests on. Slow rebound at the wheel must be AC's own DAMP_REBOUND.
    for (auto axle = std::size_t{0}; axle < 2; axle++)
    {
        const auto& corner = built->corners[axles[axle].index];
        const auto kinematics = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0, 0.0);
        REQUIRE(kinematics.has_value());

        const auto ratio = std::abs(kinematics->motionRatio);
        const auto shaftSlope = -corner.damper.points[1].y / -corner.damper.points[1].x;

        CHECK(std::abs(shaftSlope * ratio * ratio - acReboundRate[axle]) < 1e-6);
        CHECK(std::abs(-corner.damper.points[1].x / ratio - acReboundKnee[axle]) < 1e-12);
    }

    // Seal friction is stated per axle and the rear's is the sourced 25 N, not the front's 107.
    CHECK(front.damperFriction > rear.damperFriction);
    CHECK(rear.damperFriction > 0.0);

    // No car states a friction velocity shape. If one ever does, every friction figure in this file
    // changes meaning and the tables above have to be re-read.
    CHECK(front.damperFrictionShape.points.empty());
    CHECK(rear.damperFrictionShape.points.empty());
}

TEST_CASE("a second operating point where the minimum rear load is still a number", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    std::printf("\n=== the pedal at which the rear stops lifting, and the same A/B run there ===\n");
    std::printf("  **Why this exists and what it is not.** At the four-wheels fixture's own 0.50 pedal\n");
    std::printf("  every arm below — including one with NO rear damping at all — puts the rear tyre at\n");
    std::printf("  exactly 0 N, so `minimum rear load` saturates and cannot separate them. This is a\n");
    std::printf("  SECOND operating point on the same manoeuvre, added so the load has somewhere to go;\n");
    std::printf("  the primary A/B above is unchanged and is still run at 0.50 and 1.00.\n");

    std::printf("\n  --- shipped car, pedal sweep ---\n");
    std::printf("    pedal   min rear load   first lift    airborne   peak pitch   peak rear ext\n");
    std::printf("              (N)             (s)         (ticks)     (mrad)         (mm)\n");

    auto subCritical = 0.0;

    for (const auto pedal : {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50})
    {
        const auto run = runBraking(built.value(), built.value(), world.value(), pedal);
        const auto airborne = run.airborneRearTicks();

        std::printf("    %5.2f   %13.2f   %10.4f   %9d   %10.4f   %13.3f\n", pedal, run.minimumRearLoad(),
                    run.timeToFirstLift(), airborne, 1000.0 * run.peakPitch(), 1000.0 * run.peakRearExtension());

        if (airborne == 0)
        {
            subCritical = pedal;
        }
    }

    if (subCritical <= 0.0)
    {
        std::printf("\n  No pedal in the sweep keeps the rear down on the shipped car. Nothing further\n");
        std::printf("  can be said at a second operating point and this case stops here.\n");
        return;
    }

    std::printf("\n  --- the arms, at %.2f pedal (the largest swept pedal the shipped car keeps down) ---\n",
                subCritical);

    const auto& shippedCurve = built->corners[rearLeft].damper;
    auto frontDead = built.value();
    frontDead.corners[0].damper = deadCurve();
    frontDead.corners[1].damper = deadCurve();

    reportHeader();
    reportMetrics("A: shipped", measure(runBraking(built.value(), built.value(), world.value(), subCritical)));
    reportMetrics("B: rebound removed",
                  measure(runBraking(built.value(), withRearDamper(built.value(), halfCurve(shippedCurve, true, false)),
                                     world.value(), subCritical)));
    reportMetrics("C: bump removed",
                  measure(runBraking(built.value(), withRearDamper(built.value(), halfCurve(shippedCurve, false, true)),
                                     world.value(), subCritical)));
    reportMetrics("D: rear viscous removed",
                  measure(runBraking(built.value(), withRearDamper(built.value(), deadCurve()), world.value(),
                                     subCritical)));
    reportMetrics("E: front viscous removed", measure(runBraking(built.value(), frontDead, world.value(), subCritical)));

    for (const auto multiplier : {0.5, 1.5})
    {
        const auto arm = withRearDamper(built.value(), scaledRebound(shippedCurve, multiplier));
        const auto label = std::string("rebound x") + std::to_string(multiplier).substr(0, 3);
        reportMetrics(label.c_str(), measure(runBraking(built.value(), arm, world.value(), subCritical)));
    }

    std::printf("\n  Read the `minRear` column here and the `airborne`/`deficit` columns above: they\n");
    std::printf("  are the same question asked where it has an answer and where it saturates.\n");
}

TEST_CASE("is the ring frequency a property of the suspension or of the braking?", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    std::printf("\n=== identical excitation, one variable: is the brake torque still there? ===\n");
    std::printf("  Both arms take the same 0.35 pedal step from the same rolling state. The HELD arm\n");
    std::printf("  keeps it down; the PULSE arm lets it go at 0.15 s, by which time the pitch step has\n");
    std::printf("  been delivered. Both are read over the SAME window, 0.25 s to 1.75 s, so the\n");
    std::printf("  excitation, the window and the car are identical and the only difference is whether\n");
    std::printf("  brake torque and its tyre longitudinal force are present while the car rings.\n");
    std::printf("  0.35 pedal is used because the shipped car keeps all four wheels down at it, so\n");
    std::printf("  neither window contains a contact loss.\n");

    const auto held = runBraking(built.value(), built.value(), world.value(), 0.35);
    const auto pulse = runBraking(built.value(), built.value(), world.value(), 0.35, false, 0.15);
    const auto dead = runBraking(built.value(), withRearDamper(built.value(), deadCurve()), world.value(), 0.35);
    const auto deadPulse =
        runBraking(built.value(), withRearDamper(built.value(), deadCurve()), world.value(), 0.35, false, 0.15);

    constexpr auto from = std::size_t{90};
    constexpr auto to = std::size_t{630};

    REQUIRE(held.samples.size() > to);
    REQUIRE(pulse.samples.size() > to);

    struct Channel
    {
        const char* name;
        std::size_t what;
    };

    const auto channels = std::array{Channel{"pitch angle", 0}, Channel{"pitch rate", 1},
                                     Channel{"rear-left wheel travel", 2}, Channel{"rear-left tyre load", 3},
                                     Channel{"front-left tyre load", 3}};

    const auto read = [&](const Run& run, const std::size_t what, const std::size_t corner)
    {
        const auto window = detrended(run.channel(what, corner), from, to);

        return std::pair{dominantFrequency(window, 0.5, 20.0), prominentMaxima(window, 30).size()};
    };

    std::printf("\n    signal                   held (Hz)  pk | pulse (Hz)  pk |  no rear damping:"
                " held (Hz)   pulse (Hz)\n");

    for (auto index = std::size_t{0}; index < channels.size(); index++)
    {
        const auto corner = index == 4 ? frontLeft : rearLeft;
        const auto heldRead = read(held, channels[index].what, corner);
        const auto pulseRead = read(pulse, channels[index].what, corner);
        const auto deadRead = read(dead, channels[index].what, corner);
        const auto deadPulseRead = read(deadPulse, channels[index].what, corner);

        std::printf("    %-24s %9.3f  %2zu | %10.3f  %2zu |  %20.3f  %11.3f\n", channels[index].name, heldRead.first,
                    heldRead.second, pulseRead.first, pulseRead.second, deadRead.first, deadPulseRead.first);
    }

    std::printf("\n    held  arm: speed %.2f -> %.2f m/s across the window, %d airborne rear ticks\n",
                held.samples[from].speed, held.samples[to - 1].speed, held.airborneRearTicks());
    std::printf("    pulse arm: speed %.2f -> %.2f m/s across the window, %d airborne rear ticks\n",
                pulse.samples[from].speed, pulse.samples[to - 1].speed, pulse.airborneRearTicks());
    std::printf("\n  A frequency read off a heavily damped signal is worth exactly as much as the\n");
    std::printf("  number of cycles in it, which is why the peak count is printed beside every one.\n");
    std::printf("  Two or fewer prominent peaks is a shape, not a frequency.\n");

    std::printf("\n  --- the pulse arm's free ring-down, pitch angle, tick by tick every 6th sample ---\n");
    std::printf("      t (s)     pitch (mrad)   pitchRate (rad/s)   rear RL load (N)   travel RL (mm)\n");
    for (auto index = from; index < to; index += 6)
    {
        const auto& sample = pulse.samples[index];
        std::printf("     %7.4f   %12.4f   %17.5f   %16.1f   %14.3f\n", sample.time, 1000.0 * sample.pitch,
                    sample.pitchRate, sample.load[rearLeft], 1000.0 * sample.wheelTravel[rearLeft]);
    }
}

TEST_CASE("what is oscillating late in the 0.50 pedal stop, and is it the suspension?", "[.rear-damper]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(gripPlate());
    REQUIRE(world.has_value());

    const auto run = runBraking(built.value(), built.value(), world.value(), 0.50);
    REQUIRE(run.samples.size() > 900);

    std::printf("\n=== the late window of the 0.50 pedal stop, where a sustained ring does exist ===\n");
    std::printf("  The first excursion is a single near-critically-damped overshoot with a wheel\n");
    std::printf("  leaving the road in it. From about 1.2 s the car rings steadily instead, and THAT\n");
    std::printf("  is the oscillation an earlier note put at ~3.3 Hz. This case asks what drives it.\n");

    constexpr auto from = std::size_t{432};
    constexpr auto to = std::size_t{864};

    struct Channel
    {
        const char* name;
        std::size_t what;
        std::size_t corner;
    };

    const auto channels =
        std::array{Channel{"pitch angle", 0, rearLeft},        Channel{"rear-left tyre load", 3, rearLeft},
                   Channel{"front-left tyre load", 3, frontLeft}, Channel{"front-left slip ratio", 6, frontLeft},
                   Channel{"rear-left slip ratio", 6, rearLeft},  Channel{"rear-left wheel travel", 2, rearLeft}};

    std::printf("\n    signal                      dominant (Hz)   prominent peaks\n");
    for (const auto& channel : channels)
    {
        const auto window = detrended(run.channel(channel.what, channel.corner), from, to);
        std::printf("    %-26s  %12.3f   %15zu\n", channel.name, dominantFrequency(window, 0.5, 20.0),
                    prominentMaxima(window, 30).size());
    }

    std::printf("\n    t (s)   speed   FL slip   RL slip   FL load   RL load   pitch     RL travel   RL shaftVel\n");
    std::printf("            (m/s)                          (N)       (N)     (mrad)      (mm)        (m/s)\n");
    for (auto index = from; index < to; index += 8)
    {
        const auto& sample = run.samples[index];
        std::printf("   %6.3f  %6.2f  %8.4f  %8.4f  %8.0f  %8.0f  %8.3f  %10.3f  %11.5f\n", sample.time, sample.speed,
                    sample.slipRatio[frontLeft], sample.slipRatio[rearLeft], sample.load[frontLeft],
                    sample.load[rearLeft], 1000.0 * sample.pitch, 1000.0 * sample.wheelTravel[rearLeft],
                    sample.shaftVelocity[rearLeft]);
    }

    std::printf("\n  Read the slip column first. A slip ratio pinned at -1.0000 is a wheel that has\n");
    std::printf("  STOPPED TURNING, so nothing about the brake or the wheel is cycling and this is not\n");
    std::printf("  a longitudinal limit cycle in the wheel speed. What oscillates is the body on its\n");
    std::printf("  springs while four locked tyres slide, and the longitudinal force each one makes is\n");
    std::printf("  its own vertical load times a sliding friction — so the load transfer feeds back\n");
    std::printf("  into itself through the geometric load path. Assists are OFF, so nothing is\n");
    std::printf("  modulating pressure either.\n");

    std::printf("\n  --- and does the REAR DAMPER set that late frequency, or its decay? ---\n");

    const auto& shippedCurve = built->corners[rearLeft].damper;
    const auto arms = std::array{std::pair{"A: shipped", built.value()},
                                 std::pair{"B: rebound removed", withRearDamper(built.value(),
                                                                                halfCurve(shippedCurve, true, false))},
                                 std::pair{"D: rear viscous removed", withRearDamper(built.value(), deadCurve())},
                                 std::pair{"rebound x1.5", withRearDamper(built.value(),
                                                                          scaledRebound(shippedCurve, 1.5))}};

    std::printf("    arm                        pitch f (Hz)  peaks   RL load f (Hz)  peaks   RL load swing (N)\n");

    for (const auto& entry : arms)
    {
        const auto armRun = runBraking(built.value(), entry.second, world.value(), 0.50);
        if (armRun.samples.size() <= to)
        {
            continue;
        }

        const auto pitchWindow = detrended(armRun.channel(0, rearLeft), from, to);
        const auto loadWindow = detrended(armRun.channel(3, rearLeft), from, to);

        auto lowest = 1e30;
        auto highest = -1e30;
        for (auto index = from; index < to; index++)
        {
            lowest = std::min(lowest, armRun.samples[index].load[rearLeft]);
            highest = std::max(highest, armRun.samples[index].load[rearLeft]);
        }

        std::printf("    %-26s %12.3f  %5zu   %14.3f  %5zu   %17.1f\n", entry.first,
                    dominantFrequency(pitchWindow, 0.5, 20.0), prominentMaxima(pitchWindow, 30).size(),
                    dominantFrequency(loadWindow, 0.5, 20.0), prominentMaxima(loadWindow, 30).size(),
                    highest - lowest);
    }

    std::printf("\n  Note what the swing column is NOT: the late ring never takes the rear near zero.\n");
    std::printf("  The wheel leaves the road on the FIRST excursion, at about 0.11 s, and is back on\n");
    std::printf("  it long before this window opens. These are two different events.\n");
}
