// A probe, not a gate. Hidden behind a dotted tag, run by hand:
//   OSR_DROOP_OUT=<dir> [OSR_DROOP_ARM=current|disabled|gap:<mm>] [OSR_DROOP_PEDALS=0.12,0.24,...]
//       ./EngineTests "[.droop-audit]"
//
// The 2026-09-08 re-audit of the REAR droop / top-out placement on the corrected 28 kN/m Golf
// (docs/droop-placement-brief.md): the rear extension-side travel budget in wheel and shaft
// coordinates, a closed-form corner-rig unload curve, a whole-car quasi-static unload through a
// stated pitch moment, the deterministic braking ladder (the characterisation's own half-pedal stop
// and its pedal siblings, nobody intervening) with every rear channel and the corner's generalised
// force split into its terms, and a release/rebound variant of the 0.83 g case. Every scenario
// writes one CSV into `OSR_DROOP_OUT`; the analysis is `scripts/droop-audit-analysis.py`, offline,
// so that this file measures and never judges.
//
// **Nothing here changes a production parameter.** The forensic arms restate the REAR droop stop on
// a copy of the car inside this process: `disabled` zeroes its rate and its viscous constant so the
// element produces nothing at any extension (the gap stays, so `validateCornerSetup` still passes
// and "in the stop's range" can still be counted); `gap:<mm>` moves the touch point on the shaft.
// The linkage's own range clamp (`CornerHardpoints::droopAngle`) is not moved by any arm. The front
// is untouched by every arm. The arms are forensic controls and not candidate Golf values.
//
// Deterministic: fixed ticks, no clock, no randomness.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bodyToWorld;
using raceengine::bringUpJolt;
using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::CornerHardpoints;
using raceengine::CornerSetup;
using raceengine::CornerSolution;
using raceengine::damperElementOf;
using raceengine::DamperSolution;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::MassComponent;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::RigidBodyState;
using raceengine::seedTyreGasPressures;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperGeometry;
using raceengine::solveElement;
using raceengine::solveSpringForce;
using raceengine::stepVehicle;
using raceengine::SuspensionState;
using raceengine::tearDownJolt;
using raceengine::validateCornerSetup;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto degrees = 180.0 / 3.14159265358979323846;
constexpr auto gravity = 9.80665;
// The Golf's axle stations, chassis frame: half the 2.638 m wheelbase each way (PublishedCars.cppm).
constexpr auto frontAxle = 1.319;
constexpr auto rearAxle = -1.319;
constexpr auto wheelbase = frontAxle - rearAxle;

// The steady-braking window every ladder member is read over: 1.5 s after the pedal lands (the
// pitch transient settles in under a second on the corrected car), half a second long, and the car
// is still moving at its end on every member (25 m/s at 1.0 g stops after 2.55 s).
constexpr auto pedalTick = 180;
constexpr auto windowFrom = 720;
constexpr auto windowTo = 900;

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

[[nodiscard]] std::string outputDirectory()
{
    const auto* env = std::getenv("OSR_DROOP_OUT");
    return env != nullptr ? std::string(env) : std::string("droop-out");
}

// The forensic arm, on the rear droop stop alone.
struct Arm
{
    std::string name = "current";
    bool disabled = false;
    double gap = 0.0;
};

[[nodiscard]] Arm parseArm()
{
    auto arm = Arm{};
    const auto* env = std::getenv("OSR_DROOP_ARM");
    if (env == nullptr || *env == '\0' || std::string(env) == "current")
    {
        return arm;
    }

    const auto text = std::string(env);
    arm.name = text;
    if (text == "disabled")
    {
        arm.disabled = true;
    }
    else if (text.rfind("gap:", 0) == 0)
    {
        arm.gap = 0.001 * std::strtod(text.c_str() + 4, nullptr);
        REQUIRE(arm.gap > 0.0);
    }
    else
    {
        FAIL("OSR_DROOP_ARM must be current, disabled or gap:<mm>");
    }

    return arm;
}

void applyArm(VehicleSetup& setup, const Arm& arm)
{
    for (const auto index : {std::size_t{2}, std::size_t{3}})
    {
        auto& corner = setup.corners[index];
        if (arm.disabled)
        {
            corner.droopStop.rate = 0.0;
            corner.droopStop.damping = 0.0;
        }
        if (arm.gap > 0.0)
        {
            corner.droopStop.gap = arm.gap;
        }
        const auto validated = validateCornerSetup(corner);
        if (!validated)
        {
            FAIL(validated.error());
        }
    }
}

// The pedal ladder, or the default that lands near 0.2 / 0.4 / 0.6 / 0.7 g, the tyre's own peak
// (0.93 g at pedal 0.36 on the shipped car, nobody intervening — the highest steady deceleration
// this fixture can hold, so it stands in for the "~1.0 g" member) and the characterisation's own
// 0.83 g half-pedal stop. Calibrated on the first run: line pressure is linear in the pedal, the
// deceleration is not (0.12 -> 0.37 g, 0.24 -> 0.66, 0.36 -> 0.93, 0.42 / 0.50 / 0.60 -> 0.85 /
// 0.83 / 0.83 on the tyre's falling side).
[[nodiscard]] std::vector<double> pedalLadder()
{
    auto pedals = std::vector<double>{};
    const auto* env = std::getenv("OSR_DROOP_PEDALS");
    auto text = std::string(env != nullptr ? env : "0.065,0.135,0.21,0.25,0.36,0.50");
    auto position = std::size_t{0};
    while (position < text.size())
    {
        const auto comma = text.find(',', position);
        const auto piece = text.substr(position, comma == std::string::npos ? std::string::npos : comma - position);
        if (!piece.empty())
        {
            pedals.push_back(std::strtod(piece.c_str(), nullptr));
        }
        if (comma == std::string::npos)
        {
            break;
        }
        position = comma + 1;
    }
    return pedals;
}

[[nodiscard]] ProvingGroundDescriptor plate(const double size)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    return descriptor;
}

// --- geometry, straight off the elements so nothing here needs a vehicle state -----------------

// The damper shaft's EXTENSION against its design length at a stated wishbone angle, positive in
// droop. Exactly `-damperShaftCompression`.
[[nodiscard]] double shaftExtensionAt(const CornerHardpoints& hardpoints, const double angle)
{
    const auto element = damperElementOf(hardpoints);
    return solveElement(hardpoints, element, angle).length - solveElement(hardpoints, element, 0.0).length;
}

// The wishbone angle at a stated shaft extension, by bisection between design and the linkage's own
// droop limit (monotonic over that range on both of this car's axles).
[[nodiscard]] double angleForExtension(const CornerHardpoints& hardpoints, const double extension)
{
    auto low = 0.0;
    auto high = hardpoints.droopAngle;
    for (auto step = 0; step < 80; step++)
    {
        const auto middle = 0.5 * (low + high);
        if (shaftExtensionAt(hardpoints, middle) < extension)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }
    return 0.5 * (low + high);
}

struct AtAngle
{
    double angle = 0.0;
    double extension = 0.0;
    double wheelTravel = 0.0;
    double travelPerAngle = 0.0;
    double lengthPerAngle = 0.0;
    double motionRatio = 0.0;
};

[[nodiscard]] AtAngle sample(const CornerHardpoints& hardpoints, const double angle)
{
    const auto solved = solveCornerWithJacobian(hardpoints, angle, 0.0);
    REQUIRE(solved.has_value());
    const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
    REQUIRE(design.has_value());
    const auto evaluated = solveElement(hardpoints, damperElementOf(hardpoints), angle);

    auto result = AtAngle{};
    result.angle = angle;
    result.extension = shaftExtensionAt(hardpoints, angle);
    result.wheelTravel = solved->wheelTravel - design->wheelTravel;
    result.travelPerAngle = solved->travelPerAngle;
    result.lengthPerAngle = evaluated.lengthPerAngle;
    result.motionRatio = solved->travelPerAngle != 0.0 ? -evaluated.lengthPerAngle / solved->travelPerAngle : 0.0;
    return result;
}

// The patch Jacobian exactly as the force pass assembles it on the geometric load path: the wheel
// centre's plus the upright's rotation crossed with the loaded-radius arm to the patch.
[[nodiscard]] glm::dvec3 patchJacobian(const CornerSetup& corner, const SuspensionState& suspension,
                                       const double penetration)
{
    const auto belowCentre = suspension.contactPatch - suspension.wheelCentre;
    const auto freeRadius = std::max(glm::length(belowCentre), 1e-12);
    const auto roadRadius = std::max(corner.hardpoints.wheelRadius - penetration, 1e-3);
    return suspension.wheelCentrePerAngle +
           glm::cross(suspension.uprightRatePerAngle, belowCentre * (roadRadius / freeRadius));
}

// The corner's generalised force split into the terms the force pass sums, each divided by the
// wheel's vertical Jacobian so it reads as an equivalent force AT THE WHEEL, newtons, positive
// toward bump (pushing the wheel up toward the body). Their sum is `generalisedForce / t'` up to the
// driveline's shaft-work term (zero on every run here: no drive torque) and the kerb path (flat
// road). `ratioVertical` is `(n_b · P) / t'`: how far the geometric path's projection of the tyre
// load is from the plain vertical one.
struct Balance
{
    double tyreVertical = 0.0;
    double tyreLongitudinal = 0.0;
    double tyreLateral = 0.0;
    double spring = 0.0;
    double damper = 0.0;
    double bumpStop = 0.0;
    double droopStop = 0.0;
    double antiRoll = 0.0;
    double unsprungWeight = 0.0;
    double total = 0.0;
    double ratioVertical = 0.0;
};

[[nodiscard]] Balance balanceOf(const VehicleSetup& setup, const CornerSetup& corner, const CornerSolution& solution,
                                const RigidBodyState& chassis, const DamperSolution& damper)
{
    const auto& suspension = solution.suspension;
    const auto travel = suspension.travelPerAngle;
    if (std::abs(travel) < 1e-12)
    {
        return {};
    }

    const auto toBody = glm::inverse(chassis.orientation);
    const auto normal = solution.patch.inContact ? solution.patch.normal : glm::dvec3(0.0, 1.0, 0.0);

    auto balance = Balance{};
    if (setup.geometricLoadPath)
    {
        const auto jacobian =
            patchJacobian(corner, suspension, solution.patch.inContact ? solution.patch.penetration : 0.0);
        balance.ratioVertical = glm::dot(toBody * normal, jacobian) / travel;
        balance.tyreVertical = solution.forces.tireVertical * balance.ratioVertical;
        balance.tyreLongitudinal =
            solution.contact.tyre.longitudinal * glm::dot(toBody * solution.contact.forward, jacobian) / travel;
        balance.tyreLateral =
            solution.contact.tyre.lateral * glm::dot(toBody * solution.contact.lateral, jacobian) / travel;
    }
    else
    {
        const auto bodyUp = chassis.orientation * glm::dvec3(0.0, 1.0, 0.0);
        balance.ratioVertical = std::max(glm::dot(normal, bodyUp), 0.0);
        balance.tyreVertical = solution.forces.tireVertical * balance.ratioVertical;
    }

    const auto ratio = damper.lengthPerAngle / travel;
    balance.spring = solution.forces.spring * ratio;
    balance.damper = solution.forces.damper * ratio;
    balance.bumpStop = solution.forces.bumpStop * ratio;
    balance.droopStop = solution.forces.droopStop * ratio;
    balance.antiRoll = solution.forces.antiRoll;
    balance.unsprungWeight = -corner.unsprungMass * gravity;
    balance.total = solution.generalisedForce / travel;
    return balance;
}

// One CSV of per-tick channels. Everything the analysis wants, nothing judged here.
struct Recorder
{
    std::FILE* file = nullptr;
    std::array<double, cornerCount> designLength{};
    std::array<double, cornerCount> clampExtension{};

    Recorder(const std::string& path, const VehicleSetup& setup)
    {
        file = std::fopen(path.c_str(), "w");
        REQUIRE(file != nullptr);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            designLength[index] = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).length;
            clampExtension[index] = shaftExtensionAt(corner.hardpoints, corner.hardpoints.droopAngle);
        }

        std::fprintf(file, "t,y,vz,ax,az,pitch,pitchRate,brake,rhF,rhR");
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto* name = cornerAbbreviation(static_cast<Corner>(index));
            for (const auto* channel :
                 {"q", "qdot", "trav", "ext", "dv", "mr", "fz", "fs", "fd", "fdv", "fb", "fdr", "fdrE", "past",
                  "clampDist", "clamped", "Q", "I", "bz", "bx", "by", "bs", "bd", "bb", "bdr", "barb", "bu", "btot",
                  "ratioV", "pen", "contact", "fx", "slip", "wspeed", "flim"})
            {
                std::fprintf(file, ",%s_%s", channel, name);
            }
        }
        std::fprintf(file, "\n");
    }

    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    ~Recorder()
    {
        if (file != nullptr)
        {
            std::fclose(file);
        }
    }

    void row(const double time, const VehicleSetup& setup, const VehicleState& state, const VehicleStep& step,
             const VehicleInput& input)
    {
        const auto& chassis = state.chassis;
        std::fprintf(file, "%.6f,%.6f,%.6f,%.5f,%.5f,%.7f,%.6f,%.4f,%.5f,%.5f", time, chassis.position.y,
                     chassis.linearVelocity.z, step.telemetry.acceleration.x, step.telemetry.acceleration.z,
                     step.telemetry.pitch, step.telemetry.pitchRate, input.brake, step.rideHeight.front,
                     step.rideHeight.rear);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            const auto& solution = step.corners[index];
            const auto& suspension = solution.suspension;
            const auto damper = solveDamperGeometry(corner, suspension);
            const auto extension = damper.length - designLength[index];
            const auto viscous = corner.damper.at(solution.damperVelocity);
            const auto ratio = std::abs(suspension.travelPerAngle) > 1e-12
                                   ? -damper.lengthPerAngle / suspension.travelPerAngle
                                   : 0.0;
            const auto past = extension - corner.droopStop.gap;
            const auto elastic = -corner.droopStop.elasticForce(past);
            const auto q = state.corners[index].wishboneAngle;
            const auto clamped = q <= corner.hardpoints.droopAngle + 1e-12 ? 1 : 0;
            const auto balance = balanceOf(setup, corner, solution, chassis, damper);

            std::fprintf(file,
                         ",%.7f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%.4f,%.6f,"
                         "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%.3f,%.5f,%.4f,%.3f",
                         q, state.corners[index].wishboneRate, suspension.wheelTravel, extension,
                         solution.damperVelocity, ratio, solution.forces.tireVertical, solution.forces.spring,
                         solution.forces.damper, viscous, solution.forces.bumpStop, solution.forces.droopStop, elastic,
                         std::max(past, 0.0), clampExtension[index] - extension, clamped, solution.generalisedForce,
                         solution.generalisedInertia, balance.tyreVertical, balance.tyreLongitudinal,
                         balance.tyreLateral, balance.spring, balance.damper, balance.bumpStop, balance.droopStop,
                         balance.antiRoll, balance.unsprungWeight, balance.total, balance.ratioVertical,
                         solution.patch.penetration, solution.patch.inContact ? 1 : 0,
                         solution.contact.tyre.longitudinal, step.telemetry.wheels[index].slipRatio,
                         state.corners[index].wheelSpeed, solution.forces.rangeLimit);
        }
        std::fprintf(file, "\n");
    }
};

// Settle the car at rest from the characterisation's 0.52 m, seed the cavity gas at ideal after the
// settle, then give it its speed.
void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double startZ, const int ticks = 2160)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, startZ);

    for (auto step = 0; step < ticks; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    seedTyreGasPressures(setup, state);

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / setup.corners.front().hardpoints.wheelRadius;
    }
}

// What one braking run did, printed as the ladder's row and the trace's event list.
struct RunSummary
{
    std::array<double, cornerCount> travelMean{};
    std::array<double, cornerCount> extensionMean{};
    std::array<double, cornerCount> fzMean{};
    std::array<double, cornerCount> pastMean{};
    std::array<double, cornerCount> droopMean{};
    std::array<double, cornerCount> droopElasticMean{};
    std::array<double, cornerCount> bumpMean{};
    std::array<double, cornerCount> clampDistMean{};
    std::array<double, cornerCount> velocityRms{};
    std::array<int, cornerCount> clampedWindow{};
    std::array<double, cornerCount> clampForceMean{};
    std::array<Balance, cornerCount> balance{};

    std::array<int, cornerCount> droopTicks{};
    std::array<int, cornerCount> rangeTicks{};
    std::array<int, cornerCount> clampedTicks{};
    std::array<double, cornerCount> clampImpulse{};
    std::array<int, cornerCount> lossTicks{};
    std::array<double, cornerCount> fzMin{1e9, 1e9, 1e9, 1e9};
    std::array<double, cornerCount> fzMinTime{};
    std::array<double, cornerCount> extensionMax{-1e9, -1e9, -1e9, -1e9};
    std::array<double, cornerCount> extensionMaxTime{};
    std::array<double, cornerCount> droopPeak{};
    std::array<double, cornerCount> firstContact{-1.0, -1.0, -1.0, -1.0};
    std::array<double, cornerCount> firstClamp{-1.0, -1.0, -1.0, -1.0};
    std::array<double, cornerCount> slipMin{};
    std::array<int, cornerCount> bumpTicks{};
    std::array<double, cornerCount> bumpPeak{};

    double decelerationMean = 0.0;
    double pitchMean = 0.0;
    double pitchPeak = 0.0;
    double pitchPeakTime = 0.0;
    double speedFrom = 0.0;
    double speedTo = 0.0;
    int windowSamples = 0;
};

// One braking run: `pedal` from `pedalTick` until `releaseTick` (or held to the end), `ticks` long.
[[nodiscard]] RunSummary brake(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world,
                               const std::string& path, const double pedal, const int ticks, const int releaseTick)
{
    settle(setup, state, world, 25.0, 800.0);
    auto recorder = Recorder(path, setup);
    auto summary = RunSummary{};

    for (auto step = 0; step < ticks; step++)
    {
        auto input = VehicleInput{};
        input.brake = step >= pedalTick && step < releaseTick ? pedal : 0.0;
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick);
        REQUIRE(stepped.has_value());
        const auto now = static_cast<double>(step) * tick;
        recorder.row(now, setup, state, stepped.value(), input);

        const auto inWindow = step >= windowFrom && step < windowTo;
        if (step == windowFrom)
        {
            summary.speedFrom = state.chassis.linearVelocity.z;
        }
        if (step == windowTo - 1)
        {
            summary.speedTo = state.chassis.linearVelocity.z;
        }
        if (inWindow)
        {
            summary.decelerationMean += -stepped->telemetry.acceleration.z / gravity;
            summary.pitchMean += stepped->telemetry.pitch;
            summary.windowSamples++;
        }
        if (step >= pedalTick && std::abs(stepped->telemetry.pitch) > std::abs(summary.pitchPeak))
        {
            summary.pitchPeak = stepped->telemetry.pitch;
            summary.pitchPeakTime = now;
        }

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            const auto& solution = stepped->corners[index];
            const auto damper = solveDamperGeometry(corner, solution.suspension);
            const auto extension = damper.length - recorder.designLength[index];
            const auto past = extension - corner.droopStop.gap;
            const auto q = state.corners[index].wishboneAngle;
            const auto clamped = q <= corner.hardpoints.droopAngle + 1e-12;
            const auto travel = solution.suspension.travelPerAngle;
            // Since 2026-09-08 later still the constraint reports its own reaction at the wheel
            // (`forces.rangeLimit`, positive toward bump); before that this column read the deficit
            // the clamp swallowed, `−Q / t'`, which is the same quantity in the same sign on the
            // build that had no reaction. `travel` still guards the balance below.
            const auto clampForce = clamped && std::abs(travel) > 1e-12 ? solution.forces.rangeLimit : 0.0;

            summary.droopTicks[index] += solution.forces.droopStop != 0.0 ? 1 : 0;
            summary.rangeTicks[index] += past > 0.0 ? 1 : 0;
            summary.clampedTicks[index] += clamped ? 1 : 0;
            summary.clampImpulse[index] += clampForce * tick;
            summary.lossTicks[index] += solution.patch.inContact ? 0 : 1;
            summary.bumpTicks[index] += solution.forces.bumpStop > 0.0 ? 1 : 0;
            summary.bumpPeak[index] = std::max(summary.bumpPeak[index], solution.forces.bumpStop);
            summary.droopPeak[index] = std::max(summary.droopPeak[index], -solution.forces.droopStop);
            summary.slipMin[index] = std::min(summary.slipMin[index], stepped->telemetry.wheels[index].slipRatio);
            if (solution.forces.tireVertical < summary.fzMin[index])
            {
                summary.fzMin[index] = solution.forces.tireVertical;
                summary.fzMinTime[index] = now;
            }
            if (extension > summary.extensionMax[index])
            {
                summary.extensionMax[index] = extension;
                summary.extensionMaxTime[index] = now;
            }
            if (summary.firstContact[index] < 0.0 && step >= pedalTick && past > 0.0)
            {
                summary.firstContact[index] = now;
            }
            if (summary.firstClamp[index] < 0.0 && step >= pedalTick && clamped)
            {
                summary.firstClamp[index] = now;
            }

            if (inWindow)
            {
                const auto balance = balanceOf(setup, corner, solution, state.chassis, damper);
                summary.travelMean[index] += solution.suspension.wheelTravel;
                summary.extensionMean[index] += extension;
                summary.fzMean[index] += solution.forces.tireVertical;
                summary.pastMean[index] += std::max(past, 0.0);
                summary.droopMean[index] += solution.forces.droopStop;
                summary.droopElasticMean[index] += -corner.droopStop.elasticForce(past);
                summary.bumpMean[index] += solution.forces.bumpStop;
                summary.clampDistMean[index] += recorder.clampExtension[index] - extension;
                summary.velocityRms[index] += solution.damperVelocity * solution.damperVelocity;
                summary.clampedWindow[index] += clamped ? 1 : 0;
                summary.clampForceMean[index] += clampForce;
                auto& sum = summary.balance[index];
                sum.tyreVertical += balance.tyreVertical;
                sum.tyreLongitudinal += balance.tyreLongitudinal;
                sum.tyreLateral += balance.tyreLateral;
                sum.spring += balance.spring;
                sum.damper += balance.damper;
                sum.bumpStop += balance.bumpStop;
                sum.droopStop += balance.droopStop;
                sum.antiRoll += balance.antiRoll;
                sum.unsprungWeight += balance.unsprungWeight;
                sum.total += balance.total;
                sum.ratioVertical += balance.ratioVertical;
            }
        }
    }

    const auto n = static_cast<double>(std::max(summary.windowSamples, 1));
    summary.decelerationMean /= n;
    summary.pitchMean /= n;
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        summary.travelMean[index] /= n;
        summary.extensionMean[index] /= n;
        summary.fzMean[index] /= n;
        summary.pastMean[index] /= n;
        summary.droopMean[index] /= n;
        summary.droopElasticMean[index] /= n;
        summary.bumpMean[index] /= n;
        summary.clampDistMean[index] /= n;
        summary.velocityRms[index] = std::sqrt(summary.velocityRms[index] / n);
        summary.clampForceMean[index] /= n;
        auto& sum = summary.balance[index];
        for (auto* value : {&sum.tyreVertical, &sum.tyreLongitudinal, &sum.tyreLateral, &sum.spring, &sum.damper,
                            &sum.bumpStop, &sum.droopStop, &sum.antiRoll, &sum.unsprungWeight, &sum.total,
                            &sum.ratioVertical})
        {
            *value /= n;
        }
    }

    return summary;
}

void printRun(const char* title, const double pedal, const RunSummary& summary)
{
    std::printf("\n---- %s: pedal %.2f -> steady %.4f g over t = 2.0-2.5 s (speed %.2f -> %.2f m/s), pitch mean %.3f deg "
                "(%.3f deg/g), pitch peak %.3f deg at %.3f s ----\n",
                title, pedal, summary.decelerationMean, summary.speedFrom, summary.speedTo, summary.pitchMean * degrees,
                summary.decelerationMean > 1e-6 ? summary.pitchMean * degrees / summary.decelerationMean : 0.0,
                summary.pitchPeak * degrees, summary.pitchPeakTime);
    std::printf("  steady window means\n");
    std::printf("  %3s %8s %8s %8s %8s %8s %8s %8s %8s %8s %6s %8s\n", "cnr", "trav mm", "ext mm", "Fz N", "past mm",
                "Fdroop N", "elastic", "Fbump N", "toClamp", "dv rms", "clmpd", "Fclamp N");
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        std::printf("  %3s %8.2f %8.2f %8.1f %8.2f %8.1f %8.1f %8.1f %8.2f %8.4f %6d %8.1f\n",
                    cornerAbbreviation(static_cast<Corner>(index)), summary.travelMean[index] * 1000.0,
                    summary.extensionMean[index] * 1000.0, summary.fzMean[index], summary.pastMean[index] * 1000.0,
                    summary.droopMean[index], summary.droopElasticMean[index], summary.bumpMean[index],
                    summary.clampDistMean[index] * 1000.0, summary.velocityRms[index], summary.clampedWindow[index],
                    summary.clampForceMean[index]);
    }
    std::printf("  steady balance at the wheel, N, positive toward bump (sum should be ~0 unless clamped)\n");
    std::printf("  %3s %8s %8s %8s %8s %8s %8s %8s %8s %8s %9s %8s\n", "cnr", "Fz.P", "Fx.P", "Fy.P", "spring",
                "damper", "bump", "droop", "arb", "-m_u g", "sum", "ratioV");
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& b = summary.balance[index];
        std::printf("  %3s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %9.2f %8.5f\n",
                    cornerAbbreviation(static_cast<Corner>(index)), b.tyreVertical, b.tyreLongitudinal, b.tyreLateral,
                    b.spring, b.damper, b.bumpStop, b.droopStop, b.antiRoll, b.unsprungWeight, b.total, b.ratioVertical);
    }
    std::printf("  whole run\n");
    std::printf("  %3s %6s %6s %6s %9s %5s %8s %7s %8s %7s %8s %8s %8s %8s %8s\n", "cnr", "droopT", "rangeT",
                "clampT", "clampNs", "lossT", "Fz min", "at s", "ext max", "at s", "Fdr pk", "1st cont", "1st clmp",
                "slipMin", "bumpT/pk");
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        std::printf("  %3s %6d %6d %6d %9.2f %5d %8.1f %7.3f %8.2f %7.3f %8.1f %8.3f %8.3f %8.4f %4d/%.0f\n",
                    cornerAbbreviation(static_cast<Corner>(index)), summary.droopTicks[index], summary.rangeTicks[index],
                    summary.clampedTicks[index], summary.clampImpulse[index], summary.lossTicks[index],
                    summary.fzMin[index], summary.fzMinTime[index], summary.extensionMax[index] * 1000.0,
                    summary.extensionMaxTime[index], summary.droopPeak[index], summary.firstContact[index],
                    summary.firstClamp[index], summary.slipMin[index], summary.bumpTicks[index],
                    summary.bumpPeak[index]);
    }
    std::fflush(stdout);
}

void printStatic(const char* title, const VehicleSetup& setup, const VehicleState& state, const VehicleStep& step)
{
    std::printf("\n==== %s ====\n", title);
    std::printf("chassis y %.4f m, pitch %.4f deg, ride height front %.4f rear %.4f m, mass %.2f kg\n",
                state.chassis.position.y, step.telemetry.pitch * degrees, step.rideHeight.front, step.rideHeight.rear,
                state.chassis.mass);
    std::printf("  %3s %8s %9s %8s %8s %8s %9s %8s %8s %8s %8s %8s\n", "cnr", "Fz N", "q rad", "trav mm", "ext mm",
                "dmpMR", "Fspr N", "Fdmp N", "Fbump N", "Fdroop N", "toDroop", "toClamp");
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        const auto& solution = step.corners[index];
        const auto damper = solveDamperGeometry(corner, solution.suspension);
        const auto design = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).length;
        const auto extension = damper.length - design;
        const auto ratio = -damper.lengthPerAngle / solution.suspension.travelPerAngle;
        const auto clamp = shaftExtensionAt(corner.hardpoints, corner.hardpoints.droopAngle);
        std::printf("  %3s %8.1f %9.5f %8.2f %8.2f %8.4f %9.1f %8.2f %8.2f %8.2f %8.2f %8.2f\n",
                    cornerAbbreviation(static_cast<Corner>(index)), solution.forces.tireVertical,
                    state.corners[index].wishboneAngle, solution.suspension.wheelTravel * 1000.0, extension * 1000.0,
                    ratio, solution.forces.spring, solution.forces.damper, solution.forces.bumpStop,
                    solution.forces.droopStop, (corner.droopStop.gap - extension) * 1000.0, (clamp - extension) * 1000.0);
    }
}

} // namespace

TEST_CASE("rear droop placement audit: the corrected 28 kN/m Golf, measured", "[.droop-audit]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    // The car as it ships, with the characterisation's one control: the cavity gas held at the ideal
    // pressure so the tyre rate does not wander with temperature over a run. `setup` is a copy and
    // the production car is unchanged.
    auto audited = built.value();
    audited.tyrePressure = false;
    // `OSR_DROOP_KINEMATIC_REACTION=off` states `VehicleSetup::kinematicReaction` off on the copy: the
    // chassis reaction before 2026-09-08 latest of all (d) (docs/chassis-kinematic-reaction-brief.md).
    if (const auto* env = std::getenv("OSR_DROOP_KINEMATIC_REACTION"); env != nullptr && std::string(env) == "off")
    {
        audited.kinematicReaction = false;
    }

    const auto arm = parseArm();
    applyArm(audited, arm);
    const auto& setup = audited;
    const auto directory = outputDirectory();

    std::printf("\n==== droop audit arm: %s (rear droop stop gap %.4f m, rate %.0f, prog %.1f, damping %.0f; front untouched) ====\n",
                arm.name.c_str(), setup.corners[2].droopStop.gap, setup.corners[2].droopStop.rate,
                setup.corners[2].droopStop.progression, setup.corners[2].droopStop.damping);
    std::printf("switches: geometricLoadPath %d drivelineReaction %d tyreThermal %d tyrePressure %d brakeThermal %d kerbContact %d frameAcceleration %d kinematicReaction %d\n",
                setup.geometricLoadPath, setup.drivelineReaction, setup.tyreThermal, setup.tyrePressure,
                setup.brakeThermal, setup.kerbContact, setup.frameAcceleration, setup.kinematicReaction);

    const auto flat = PhysicsWorld::create(generateProvingGround(plate(1600.0)).value());
    REQUIRE(flat.has_value());

    auto state = VehicleState{};

    // --- 1. the extension-side travel budget, wheel and shaft coordinates ---------------------------
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, flat.value(), tick);
        REQUIRE(stepped.has_value());
        printStatic("static, settled 6 s at rest, gas seeded at ideal", setup, state, stepped.value());

        std::printf("\n==== extension-side travel budget from the design position (q = 0 is static ride) ====\n");
        for (const auto index : {std::size_t{0}, std::size_t{2}})
        {
            const auto& corner = setup.corners[index];
            const auto& hardpoints = corner.hardpoints;
            const auto& stop = corner.droopStop;
            const auto design = sample(hardpoints, 0.0);
            const auto touch = sample(hardpoints, angleForExtension(hardpoints, stop.gap));
            const auto clamp = sample(hardpoints, hardpoints.droopAngle);
            const auto usable = clamp.extension - stop.gap;
            const auto springAtDesign = solveSpringForce(corner, solveCornerWithJacobian(hardpoints, 0.0, 0.0).value());
            const auto springAtClamp = solveSpringForce(corner, solveCornerWithJacobian(hardpoints, hardpoints.droopAngle, 0.0).value());
            const auto unsprungWeight = corner.unsprungMass * gravity;
            const auto settledExtension = solveDamperGeometry(corner, stepped->corners[index].suspension).length -
                                          solveElement(hardpoints, damperElementOf(hardpoints), 0.0).length;

            std::printf("\n  --- %s (index %zu) ---\n", index == 0 ? "front (reference, untouched by every arm)" : "REAR",
                        index);
            std::printf("    settled static: wheel travel %+.3f mm, shaft extension %+.3f mm, Fz %.1f N, spring %.1f N on the axis\n",
                        stepped->corners[index].suspension.wheelTravel * 1000.0, settledExtension * 1000.0,
                        stepped->corners[index].forces.tireVertical, stepped->corners[index].forces.spring);
            std::printf("    design (q = 0): damper MR %.4f (dL/dq %.5f m/rad, dz/dq %.5f m/rad); spring axis force %.1f N\n",
                        design.motionRatio, design.lengthPerAngle, design.travelPerAngle, springAtDesign.force);
            std::printf("    droop-stop touch: shaft extension %.2f mm (= gap) at q %.5f rad, wheel travel %.2f mm, MR %.4f\n",
                        touch.extension * 1000.0, touch.angle, touch.wheelTravel * 1000.0, touch.motionRatio);
            std::printf("    linkage clamp (authored droopAngle %.5f rad): shaft extension %.2f mm, wheel travel %.2f mm, MR %.4f\n",
                        hardpoints.droopAngle, clamp.extension * 1000.0, clamp.wheelTravel * 1000.0, clamp.motionRatio);
            std::printf("    static -> touch: %.2f mm shaft / %.2f mm wheel; static -> clamp: %.2f / %.2f; touch -> clamp: %.2f / %.2f\n",
                        stop.gap * 1000.0, -touch.wheelTravel * 1000.0, clamp.extension * 1000.0, -clamp.wheelTravel * 1000.0,
                        usable * 1000.0, (touch.wheelTravel - clamp.wheelTravel) * 1000.0);
            std::printf("    spring axis force at the clamp %.1f N (%.1f N at the wheel through MR %.4f); unsprung weight %.1f N\n",
                        springAtClamp.force, springAtClamp.force * clamp.motionRatio, clamp.motionRatio, unsprungWeight);
            if (stop.rate > 0.0)
            {
                std::printf("    stop elastic reach at the clamp %.1f N (shaft) = %.1f N at the wheel; law rate %.0f prog %.1f over gap %.4f\n",
                            stop.elasticForce(usable), stop.elasticForce(usable) * clamp.motionRatio, stop.rate,
                            stop.progression, stop.gap);
                std::printf("      x past touch (mm) -> elastic N:");
                for (const auto millimetres : {1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.57, 15.0, 20.0})
                {
                    std::printf(" %.2f:%.0f%s", millimetres, stop.elasticForce(0.001 * millimetres),
                                0.001 * millimetres > usable + 1e-9 ? "(past clamp)" : "");
                }
                std::printf("\n");
            }
            else
            {
                std::printf("    stop DISABLED on this arm (rate 0, damping 0): no elastic reach\n");
            }

            std::printf("\n      q (rad)   ext mm   wheel mm       MR   dL/dq m/rad   dz/dq m/rad\n");
            for (auto step = 0; step <= 12; step++)
            {
                const auto angle = hardpoints.droopAngle * static_cast<double>(step) / 12.0;
                const auto point = sample(hardpoints, angle);
                std::printf("      %8.5f %8.2f %10.2f %8.4f %13.5f %13.5f\n", angle, point.extension * 1000.0,
                            point.wheelTravel * 1000.0, point.motionRatio, point.lengthPerAngle, point.travelPerAngle);
            }
        }
        std::fflush(stdout);
    }

    // --- 2. the corner-rig unload curve, closed form: body held level, the wheel load that balances
    //        the corner at each extension with zero velocity, through the force pass's own terms -------
    {
        std::printf("\n==== corner-rig unload curve (closed form, body level, q̇ = 0): rear ====\n");
        std::printf("  Fz = [m_u g dz/dq - (F_spring + F_droop) dL/dq] / (P_y) per the geometric load path; the spring-alone\n");
        std::printf("  column drops the stop. The clamp row is the last reachable position.\n");
        auto* file = std::fopen((directory + "/rig_rear.csv").c_str(), "w");
        REQUIRE(file != nullptr);
        std::fprintf(file, "ext,trav,mr,fspring,fdroop,fz,fzSpringOnly,rate,clampDist\n");
        const auto& corner = setup.corners[2];
        const auto& hardpoints = corner.hardpoints;
        const auto clampExtension = shaftExtensionAt(hardpoints, hardpoints.droopAngle);
        std::printf("  %8s %9s %8s %9s %9s %9s %10s %10s %9s\n", "ext mm", "wheel mm", "MR", "Fspr N", "Fdroop N",
                    "Fz N", "Fz noStop", "kN/m", "toClamp");

        auto previousFz = 0.0;
        auto previousTravel = 0.0;
        for (auto row = 0; row <= 106; row++)
        {
            const auto extension = std::min(0.0005 * static_cast<double>(row), clampExtension);
            const auto angle = angleForExtension(hardpoints, extension);
            const auto solved = solveCornerWithJacobian(hardpoints, angle, 0.0);
            REQUIRE(solved.has_value());
            const auto point = sample(hardpoints, angle);
            const auto spring = solveSpringForce(corner, solved.value()).force;
            const auto droop = -corner.droopStop.elasticForce(extension - corner.droopStop.gap);

            auto penetration = 0.0;
            auto fz = 0.0;
            auto fzSpringOnly = 0.0;
            for (auto pass = 0; pass < 3; pass++)
            {
                const auto jacobian = setup.geometricLoadPath ? patchJacobian(corner, solved.value(), penetration)
                                                              : glm::dvec3(0.0, point.travelPerAngle, 0.0);
                const auto py = jacobian.y;
                fz = (corner.unsprungMass * gravity * point.travelPerAngle - (spring + droop) * point.lengthPerAngle) / py;
                fzSpringOnly = (corner.unsprungMass * gravity * point.travelPerAngle - spring * point.lengthPerAngle) / py;
                penetration = std::max(fz, 0.0) / corner.tireVerticalRate;
            }

            const auto rate = row > 0 && std::abs(point.wheelTravel - previousTravel) > 1e-12
                                  ? (fz - previousFz) / (point.wheelTravel - previousTravel)
                                  : 0.0;
            previousFz = fz;
            previousTravel = point.wheelTravel;

            std::fprintf(file, "%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f\n", extension, point.wheelTravel,
                         point.motionRatio, spring, droop, fz, fzSpringOnly, rate, clampExtension - extension);
            if (row % 4 == 0 || extension >= clampExtension - 1e-12 || std::abs(extension - corner.droopStop.gap) < 1e-9)
            {
                std::printf("  %8.2f %9.2f %8.4f %9.1f %9.1f %9.1f %10.1f %10.1f %9.2f%s\n", extension * 1000.0,
                            point.wheelTravel * 1000.0, point.motionRatio, spring, droop, fz, fzSpringOnly,
                            0.001 * rate, (clampExtension - extension) * 1000.0,
                            extension >= clampExtension - 1e-12 ? "  <- linkage clamp" : "");
            }
            if (extension >= clampExtension - 1e-12)
            {
                break;
            }
        }
        std::fclose(file);
        std::fflush(stdout);
    }

    // --- 3. the whole-car quasi-static unload: a pitch moment through a stated mass 8 m ahead of the
    //        front axle (rear unload m g d / L, front load m g (1 + d / L)), settled, no controller -----
    {
        constexpr auto arm_ = 8.0;
        std::printf("\n==== whole-car quasi-static rear unload: mass at z = front axle + %.1f m, settled 7 s then the last second averaged ====\n", arm_);
        std::printf("  rear unload per kg %.2f N (m g d / L), front load per kg %.2f N\n", gravity * arm_ / wheelbase,
                    gravity * (1.0 + arm_ / wheelbase));
        auto* file = std::fopen((directory + "/unload_rear.csv").c_str(), "w");
        REQUIRE(file != nullptr);
        std::fprintf(file, "kg,rearUnload,fzRL,fzRR,fzFL,travRL,extRL,fsRL,fdrRL,fdrElasticRL,clampDistRL,clampedTicks,QRL,"
                           "pitch,y,rhF,rhR,travFL,fbFL,bzRL,bxRL,bsRL,bdRL,bdrRL,buRL,btotRL,ratioVRL,tyreSum,weight\n");
        std::printf("  %5s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %6s %9s %8s %8s %8s\n", "kg", "unload N", "FzRL N",
                    "FzFL N", "travRL", "extRL", "FsprRL", "FdrRL", "elastic", "toClamp", "Qwheel", "clmpT", "pitch",
                    "travFL", "FbumpFL", "sumFz-W");

        // To 160 kg: the rear wheels hang from 150 kg on the shipped car, and past 190 kg the moment
        // tips the car over its front wheels, which is no longer a suspension measurement.
        for (auto added = 0.0; added <= 160.0 + 1e-9; added += 10.0)
        {
            auto loaded = setup;
            if (added > 0.0)
            {
                loaded.sprung.push_back(MassComponent{.mass = added,
                                                      .centre = glm::dvec3(0.0, 0.50, frontAxle + arm_),
                                                      .inertia = glm::dmat3(0.0)});
            }

            // Settled 7 s, then every quantity is the MEAN of the last second: a corner resting on
            // the linkage clamp is a velocity clamp re-applied every tick, so a single sample of it
            // reads one phase of a tick-rate alternation rather than the state.
            settle(loaded, state, flat.value(), 0.0, 800.0, 2520);
            const auto& corner = loaded.corners[2];
            const auto design = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).length;
            const auto clampExtension = shaftExtensionAt(corner.hardpoints, corner.hardpoints.droopAngle);
            const auto unload = added * gravity * arm_ / wheelbase;

            auto fzRL = 0.0, fzRR = 0.0, fzFL = 0.0, fzFR = 0.0, travRL = 0.0, extRL = 0.0, fsRL = 0.0, fdrRL = 0.0;
            auto elasticRL = 0.0, qRL = 0.0, pitch = 0.0, y = 0.0, rhF = 0.0, rhR = 0.0, travFL = 0.0, fbFL = 0.0;
            auto clampedTicks = 0;
            auto sum = Balance{};
            constexpr auto window = 360;
            for (auto step = 0; step < window; step++)
            {
                const auto stepped = stepVehicle(loaded, state, VehicleInput{}, noDriveTorque, flat.value(), tick);
                REQUIRE(stepped.has_value());
                const auto& solution = stepped->corners[2];
                const auto damper = solveDamperGeometry(corner, solution.suspension);
                const auto extension = damper.length - design;
                const auto past = extension - corner.droopStop.gap;
                const auto balance = balanceOf(loaded, corner, solution, state.chassis, damper);

                fzRL += solution.forces.tireVertical;
                fzRR += stepped->corners[3].forces.tireVertical;
                fzFL += stepped->corners[0].forces.tireVertical;
                fzFR += stepped->corners[1].forces.tireVertical;
                travRL += solution.suspension.wheelTravel;
                extRL += extension;
                fsRL += solution.forces.spring;
                fdrRL += solution.forces.droopStop;
                elasticRL += -corner.droopStop.elasticForce(past);
                qRL += solution.generalisedForce;
                pitch += stepped->telemetry.pitch;
                y += state.chassis.position.y;
                rhF += stepped->rideHeight.front;
                rhR += stepped->rideHeight.rear;
                travFL += stepped->corners[0].suspension.wheelTravel;
                fbFL += stepped->corners[0].forces.bumpStop;
                clampedTicks += state.corners[2].wishboneAngle <= corner.hardpoints.droopAngle + 1e-12 ? 1 : 0;
                sum.tyreVertical += balance.tyreVertical;
                sum.tyreLongitudinal += balance.tyreLongitudinal;
                sum.spring += balance.spring;
                sum.damper += balance.damper;
                sum.droopStop += balance.droopStop;
                sum.unsprungWeight += balance.unsprungWeight;
                sum.total += balance.total;
                sum.ratioVertical += balance.ratioVertical;
            }
            for (auto* value : {&fzRL, &fzRR, &fzFL, &fzFR, &travRL, &extRL, &fsRL, &fdrRL, &elasticRL, &qRL, &pitch, &y,
                                &rhF, &rhR, &travFL, &fbFL, &sum.tyreVertical, &sum.tyreLongitudinal, &sum.spring,
                                &sum.damper, &sum.droopStop, &sum.unsprungWeight, &sum.total, &sum.ratioVertical})
            {
                *value /= static_cast<double>(window);
            }
            const auto weight = state.chassis.mass * gravity;
            const auto tyreSum = fzFL + fzFR + fzRL + fzRR;

            std::fprintf(file, "%.1f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%.3f,%.3f,%.6f,%d,%.4f,%.7f,%.6f,%.5f,%.5f,%.6f,%.3f,"
                               "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.3f,%.3f\n",
                         added, unload, fzRL, fzRR, fzFL, travRL, extRL, fsRL, fdrRL, elasticRL, clampExtension - extRL,
                         clampedTicks, qRL, pitch, y, rhF, rhR, travFL, fbFL, sum.tyreVertical, sum.tyreLongitudinal,
                         sum.spring, sum.damper, sum.droopStop, sum.unsprungWeight, sum.total, sum.ratioVertical, tyreSum,
                         weight);
            std::printf("  %5.0f %8.1f %8.1f %8.1f %8.2f %8.2f %8.1f %8.1f %8.1f %8.2f %8.1f %6d %9.3f %8.2f %8.1f %8.1f\n",
                        added, unload, fzRL, fzFL, travRL * 1000.0, extRL * 1000.0, fsRL, fdrRL, elasticRL,
                        (clampExtension - extRL) * 1000.0, sum.total, clampedTicks, pitch * degrees, travFL * 1000.0,
                        fbFL, tyreSum - weight);
            std::fflush(stdout);
        }
        std::fclose(file);
    }

    // --- 4. the braking ladder: the characterisation's half-pedal stop from 25 m/s and its pedal
    //        siblings, nobody intervening, pedal held to the end -------------------------------------
    for (const auto pedal : pedalLadder())
    {
        auto name = std::string("/brake_p") + std::to_string(static_cast<int>(std::lround(pedal * 100.0))) + ".csv";
        const auto summary = brake(setup, state, flat.value(), directory + name, pedal, 1440, 1 << 30);
        printRun("ladder", pedal, summary);
    }

    // --- 5. the 0.83 g case with a release at t = 2.5 s, for the release and the rebound -----------
    {
        const auto summary = brake(setup, state, flat.value(), directory + "/brake_p50_release.csv", 0.5, 1800, windowTo);
        printRun("release at 2.5 s", 0.5, summary);
    }

    std::printf("\nwrote CSVs to %s\n", directory.c_str());
}
