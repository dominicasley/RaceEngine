// A probe, not a gate. Hidden behind a dotted tag, run by hand:
// `OSR_CHAR_OUT=<dir> ./EngineTests "[.suspension-characterisation]"`.
//
// The 2026-09-08 suspension characterisation: the shipped Golf, unchanged, put through a static
// settle, small-signal heave / pitch / roll ring-downs, a wheel-hop impulse per axle, single-wheel
// bumps and dips of stated size, a quasi-static axle-load sweep towards the bump stops, a
// deterministic rough road, a half-pedal stop and a steering step. Every scenario writes one CSV of
// per-tick channels into `OSR_CHAR_OUT`; the analysis is done offline so that this file only
// measures and never judges. **Nothing here changes a production parameter**: every setup edit is a
// fixture load (an added mass over an axle) on a copy of the car.
//
// Deterministic: fixed ticks, no clock, no randomness — the rough road is a stated sum of sines.
//
// **One experimental knob, and it is read here and nowhere else**: `OSR_CHAR_REAR_WHEEL_RATE=<N/m>`
// restates the rear corners' spring on the fixture's copy so that the wheel rate at the design
// position is the number given (`docs/rear-spring-sensitivity-brief.md`, 2026-09-08). The free length
// is re-solved from the static sprung load the shipped spring carries, so q = 0 stays the static
// position and every shaft-referred stop gap is exactly what it was. Unset is the shipped car, and
// production data is never touched: the arm exists only inside this process.
//
// **A second knob, the same way** (`docs/droop-placement-brief.md`, 2026-09-08): `OSR_CHAR_DROOP_ARM`
// restates the REAR droop stop on the fixture's copy — `disabled` zeroes its rate and viscous
// constant (the gap stays so validation passes), `gap:<mm>` moves its touch point on the shaft. The
// linkage clamp and the front are untouched. Forensic controls, not candidate Golf values.

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

using raceengine::angularVelocity;
using raceengine::AssistSensors;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::Corner;
using raceengine::CornerSide;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::damperElementOf;
using raceengine::defaultSurfaceMaterials;
using raceengine::DrivelineState;
using raceengine::frontLeft;
using raceengine::frontRight;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::golfGtiMk7Driveline;
using raceengine::MassComponent;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::seedTyreGasPressures;
using raceengine::setAngularVelocity;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperKinematics;
using raceengine::solveElement;
using raceengine::solveSpringKinematics;
using raceengine::springFreeLengthForLoad;
using raceengine::springElementOf;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::TractionMode;
using raceengine::updateAssists;
using raceengine::validateCornerSetup;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto degrees = 180.0 / 3.14159265358979323846;
constexpr auto pi = 3.14159265358979323846;
constexpr auto gravity = 9.80665;
// The Golf's axle stations, chassis frame: half the 2.638 m wheelbase each way (PublishedCars.cppm).
constexpr auto frontAxle = 1.319;
constexpr auto rearAxle = -1.319;

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
    const auto* env = std::getenv("OSR_CHAR_OUT");
    return env != nullptr ? std::string(env) : std::string("char-out");
}

// The experimental rear wheel rate, N/m, or 0 for the shipped car.
[[nodiscard]] double rearWheelRateArm()
{
    const auto* env = std::getenv("OSR_CHAR_REAR_WHEEL_RATE");
    return env != nullptr && *env != '\0' ? std::strtod(env, nullptr) : 0.0;
}

// The chassis's acceleration in the corner's equation (docs/frame-acceleration-brief.md, 2026-09-08
// latest of all): `OSR_CHAR_FRAME_ACCELERATION=off` states `VehicleSetup::frameAcceleration` off on
// the copy, which is the build before that change; unset is the car's own, on.
[[nodiscard]] bool frameAccelerationOff()
{
    const auto* env = std::getenv("OSR_CHAR_FRAME_ACCELERATION");
    return env != nullptr && std::string(env) == "off";
}

// The corner's configuration-dependent inertia term (docs/nonlinear-geometry-brief.md, 2026-09-08
// latest of all (c)): `OSR_CHAR_NONLINEAR_GEOMETRY=off` states `VehicleSetup::nonlinearGeometry` off
// on the copy, which is the build before that change; unset is the car's own, on.
[[nodiscard]] bool nonlinearGeometryOff()
{
    const auto* env = std::getenv("OSR_CHAR_NONLINEAR_GEOMETRY");
    return env != nullptr && std::string(env) == "off";
}

// The chassis's side of the unsprung masses' acceleration (docs/chassis-kinematic-reaction-brief.md,
// 2026-09-08 latest of all (d)): `OSR_CHAR_KINEMATIC_REACTION=off` states `VehicleSetup::kinematicReaction`
// off on the copy, which is the build before that change; unset is the car's own, on.
[[nodiscard]] bool kinematicReactionOff()
{
    const auto* env = std::getenv("OSR_CHAR_KINEMATIC_REACTION");
    return env != nullptr && std::string(env) == "off";
}

// The rear droop-stop arm, applied to a copy: "" for the shipped stop.
[[nodiscard]] std::string rearDroopArm()
{
    const auto* env = std::getenv("OSR_CHAR_DROOP_ARM");
    return env != nullptr && std::string(env) != "current" ? std::string(env) : std::string{};
}

void stateRearDroopArm(VehicleSetup& setup, const std::string& arm)
{
    for (const auto index : {std::size_t{2}, std::size_t{3}})
    {
        auto& corner = setup.corners[index];
        if (arm == "disabled")
        {
            corner.droopStop.rate = 0.0;
            corner.droopStop.damping = 0.0;
        }
        else if (arm.rfind("gap:", 0) == 0)
        {
            corner.droopStop.gap = 0.001 * std::strtod(arm.c_str() + 4, nullptr);
        }
        else
        {
            FAIL("OSR_CHAR_DROOP_ARM must be current, disabled or gap:<mm>");
        }
        REQUIRE(validateCornerSetup(corner).has_value());
    }
    std::printf("  droop arm %s: rear droop stop gap %.4f m rate %.0f damping %.0f (front untouched)\n", arm.c_str(),
                setup.corners[2].droopStop.gap, setup.corners[2].droopStop.rate, setup.corners[2].droopStop.damping);
}

// Restate the rear spring so that its wheel rate at the design position is `wheelRate`. The quantity
// swept is the EFFECTIVE WHEEL RATE: the spring element's own rate is `wheelRate / MR^2` through the
// linkage's current motion ratio, which is reported alongside and is a property of the placed rear
// geometry rather than Golf data. The static sprung load is recovered from the shipped spring's own
// preload rather than restated, and the free length is re-solved from it, so the static position
// (q = 0), the ride height and both shaft-referred stop gaps are bit-for-bit what they were.
void stateRearWheelRate(VehicleSetup& setup, const double wheelRate)
{
    for (const auto index : {std::size_t{2}, std::size_t{3}})
    {
        auto& corner = setup.corners[index];
        const auto spring = solveSpringKinematics(corner.hardpoints, springElementOf(corner.hardpoints), 0.0, 0.0);
        REQUIRE(spring.has_value());
        const auto ratio = std::abs(spring->motionRatio);

        const auto sprungLoad = (corner.springFreeLength - spring->length) * corner.springRate * ratio;
        const auto shippedRate = corner.springRate;

        corner.springRate = wheelRate / (ratio * ratio);
        const auto restLength = springFreeLengthForLoad(corner, sprungLoad);
        REQUIRE(restLength.has_value());
        corner.springFreeLength = restLength.value();
        REQUIRE(validateCornerSetup(corner).has_value());

        std::printf("  arm %s: wheel rate %.1f N/m -> spring element %.1f N/m (was %.1f) at MR %.4f; static sprung load %.1f N, "
                    "free length %.4f m, static compression %.2f mm\n",
                    cornerAbbreviation(static_cast<Corner>(index)), wheelRate, corner.springRate, shippedRate, ratio,
                    sprungLoad, corner.springFreeLength, (corner.springFreeLength - spring->length) * 1000.0);
    }
}

// A flat plate, the settle-and-ring-down surface.
[[nodiscard]] ProvingGroundDescriptor plate(const double size)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    return descriptor;
}

// A heightfield from a stated function, on stated column (x) and row (z) positions, wound exactly
// as the engine's own generator winds its grid so the front face points up.
using HeightFunction = std::function<double(double, double)>;

[[nodiscard]] SurfaceMesh heightfield(const HeightFunction& height, const std::vector<double>& xs,
                                      const std::vector<double>& zs)
{
    auto mesh = SurfaceMesh{};
    mesh.materials = defaultSurfaceMaterials();
    mesh.vertices.reserve(xs.size() * zs.size());

    for (const auto z : zs)
    {
        for (const auto x : xs)
        {
            mesh.vertices.emplace_back(x, height(x, z), z);
        }
    }

    const auto rowStride = xs.size();
    for (auto row = std::size_t{0}; row + 1 < zs.size(); row++)
    {
        for (auto column = std::size_t{0}; column + 1 < xs.size(); column++)
        {
            const auto corner = row * rowStride + column;
            const auto a = static_cast<std::uint32_t>(corner);
            const auto b = static_cast<std::uint32_t>(corner + 1);
            const auto c = static_cast<std::uint32_t>(corner + rowStride);
            const auto d = static_cast<std::uint32_t>(corner + rowStride + 1);

            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
            mesh.surfaces.push_back(0);
            mesh.surfaces.push_back(0);
        }
    }

    return mesh;
}

// Row positions: fine where the road does something, coarse where it does not.
[[nodiscard]] std::vector<double> rows(const double from, const double to, const double fineFrom, const double fineTo,
                                       const double coarse, const double fine)
{
    auto zs = std::vector<double>{};
    auto z = from;
    while (z < to)
    {
        zs.push_back(z);
        z += (z >= fineFrom - coarse && z < fineTo) ? fine : coarse;
    }
    zs.push_back(to);

    return zs;
}

// The left track band (the car's +x) and the right one, with the flat gutter between. The tyre is
// 0.225 wide and rides at |x| = 0.77 (front) / 0.758 (rear); the band is flat under it and tapers
// over the 0.25 m outside its edges.
const auto trackColumns = std::vector<double>{-3.0, -1.35, -1.10, -0.45, -0.20, 0.20, 0.45, 1.10, 1.35, 3.0};

[[nodiscard]] double smoothstep(const double edge0, const double edge1, const double x)
{
    const auto t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// A single versine hump (or dip, negative amplitude) of length `length` starting at `start`.
[[nodiscard]] double versine(const double z, const double start, const double length, const double amplitude)
{
    if (z < start || z > start + length)
    {
        return 0.0;
    }

    const auto s = std::sin(pi * (z - start) / length);
    return amplitude * s * s;
}

// One CSV of per-tick channels. Everything the analysis wants, nothing judged here.
struct Recorder
{
    std::FILE* file = nullptr;
    std::array<double, cornerCount> designLength{};

    Recorder(const std::string& path, const VehicleSetup& setup)
    {
        file = std::fopen(path.c_str(), "w");
        REQUIRE(file != nullptr);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            designLength[index] = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).length;
        }

        std::fprintf(file, "t,x,y,z,vx,vy,vz,pitch,roll,pitchRate,rollRate,vfront,vrear,ax,ay,az,steer,brake");
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto* name = cornerAbbreviation(static_cast<Corner>(index));
            std::fprintf(file,
                         ",q_%s,qdot_%s,trav_%s,shaft_%s,dv_%s,fs_%s,fd_%s,fdv_%s,fb_%s,fdr_%s,farb_%s,fz_%s,pen_%s,"
                         "contact_%s,mr_%s,tpa_%s,road_%s,wy_%s",
                         name, name, name, name, name, name, name, name, name, name, name, name, name, name, name,
                         name, name, name);
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
             const VehicleInput& input, const HeightFunction& road)
    {
        const auto& chassis = state.chassis;
        const auto omega = angularVelocity(chassis);

        // The body points above the two axles at the centre of mass's height, so a front and a
        // rear "seat" acceleration can be differenced offline.
        const auto axleFront = chassis.orientation * glm::dvec3(0.0, 0.0, frontAxle - chassis.centreOfMass.z);
        const auto axleRear = chassis.orientation * glm::dvec3(0.0, 0.0, rearAxle - chassis.centreOfMass.z);
        const auto vFront = (chassis.linearVelocity + glm::cross(omega, axleFront)).y;
        const auto vRear = (chassis.linearVelocity + glm::cross(omega, axleRear)).y;

        std::fprintf(file, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.7f,%.7f,%.6f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f,%.4f,%.4f",
                     time, chassis.position.x, chassis.position.y, chassis.position.z, chassis.linearVelocity.x,
                     chassis.linearVelocity.y, chassis.linearVelocity.z, step.telemetry.pitch, step.telemetry.roll,
                     step.telemetry.pitchRate, step.telemetry.rollRate, vFront, vRear, step.telemetry.acceleration.x,
                     step.telemetry.acceleration.y, step.telemetry.acceleration.z, input.steering, input.brake);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            const auto& solution = step.corners[index];
            const auto& suspension = solution.suspension;
            const auto damper = raceengine::solveDamperGeometry(corner, suspension);
            const auto shaft = designLength[index] - damper.length;
            const auto viscous = corner.damper.at(solution.damperVelocity);
            const auto ratio = std::abs(suspension.travelPerAngle) > 1e-12 ? damper.lengthPerAngle / suspension.travelPerAngle : 0.0;
            const auto wheelWorld = raceengine::bodyToWorld(chassis, suspension.wheelCentre);

            std::fprintf(file, ",%.7f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%d,%.6f,%.6f,%.6f,%.6f",
                         state.corners[index].wishboneAngle, state.corners[index].wishboneRate,
                         suspension.wheelTravel, shaft, solution.damperVelocity, solution.forces.spring,
                         solution.forces.damper, viscous, solution.forces.bumpStop, solution.forces.droopStop,
                         solution.forces.antiRoll, solution.forces.tireVertical, solution.patch.penetration,
                         solution.patch.inContact ? 1 : 0, ratio, suspension.travelPerAngle,
                         road(wheelWorld.x, wheelWorld.z), wheelWorld.y);
        }
        std::fprintf(file, "\n");
    }
};

const HeightFunction flatRoad = [](double, double) { return 0.0; };

// Settle the car at rest, then state the cavity gas at its ideal pressure — after the settle, which
// is the order the repaired `[.damper-friction]` fixture measured to be the right one — and give it
// its speed.
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

// A proportional-integral speed hold on the front wheels, the shape `BodyAttitudeProbe` uses, so a
// scenario at "20 m/s" is at 20 m/s for its whole length rather than coasting down.
struct SpeedHold
{
    double target = 0.0;
    double integral = 0.0;

    [[nodiscard]] std::array<double, cornerCount> torques(const VehicleSetup& setup, const VehicleState& state)
    {
        if (target <= 0.0)
        {
            return noDriveTorque;
        }

        const auto error = target - glm::length(state.chassis.linearVelocity);
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                              setup.corners.front().hardpoints.wheelRadius / 2.0;

        return {perWheel, perWheel, 0.0, 0.0};
    }
};

// Run `ticks` steps of a scenario, recording every one.
void run(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, Recorder& recorder,
         const HeightFunction& road, const int ticks, SpeedHold& hold,
         const std::function<VehicleInput(int)>& inputAt = [](int) { return VehicleInput{}; })
{
    for (auto step = 0; step < ticks; step++)
    {
        const auto input = inputAt(step);
        const auto stepped = stepVehicle(setup, state, input, hold.torques(setup, state), world, tick);
        REQUIRE(stepped.has_value());
        recorder.row(step * tick, setup, state, stepped.value(), input, road);
    }
}

// The wheel travel at which a shaft-referred stop gap is reached, by bisection on the linkage.
[[nodiscard]] double wheelTravelAtShaft(const raceengine::CornerSetup& corner, const double shaftCompression)
{
    const auto& hardpoints = corner.hardpoints;
    const auto element = damperElementOf(hardpoints);
    const auto design = solveElement(hardpoints, element, 0.0).length;

    auto low = hardpoints.droopAngle;
    auto high = hardpoints.bumpAngle;
    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto mid = 0.5 * (low + high);
        const auto compression = design - solveElement(hardpoints, element, mid).length;
        if (compression < shaftCompression)
        {
            low = mid;
        }
        else
        {
            high = mid;
        }
    }

    const auto solved = raceengine::solveCorner(hardpoints, 0.5 * (low + high), 0.0);
    return solved ? solved->wheelTravel : 0.0;
}

// The static table, printed and written.
void printStatic(const char* title, const VehicleSetup& setup, const VehicleState& state, const VehicleStep& step)
{
    std::printf("\n==== %s ====\n", title);
    std::printf("chassis y %.4f m, pitch %.4f deg, roll %.4f deg, ride height front %.4f rear %.4f m\n",
                state.chassis.position.y, step.telemetry.pitch * degrees, step.telemetry.roll * degrees,
                step.rideHeight.front, step.rideHeight.rear);
    std::printf("  %3s %8s %9s %8s %8s %8s %8s %8s %9s %8s %8s %8s %8s %8s %8s %8s\n", "cnr", "Fz N", "q rad",
                "trav mm", "shaft mm", "sprMR", "dmpMR", "sprX mm", "Fspr N", "Fdmp N", "Ffric N", "Fbump N",
                "Fdroop N", "Farb N", "toBump", "toDroop");

    auto totalFz = 0.0;
    auto frontFz = 0.0;
    auto rearFz = 0.0;
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        const auto& solution = step.corners[index];
        const auto& suspension = solution.suspension;
        const auto q = state.corners[index].wishboneAngle;

        const auto spring = solveSpringKinematics(corner.hardpoints, springElementOf(corner.hardpoints), q, 0.0);
        const auto damper = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), q, 0.0);
        REQUIRE(spring.has_value());
        REQUIRE(damper.has_value());

        const auto design = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).length;
        const auto shaft = design - damper->length;
        const auto viscous = corner.damper.at(solution.damperVelocity);
        const auto toBump = wheelTravelAtShaft(corner, corner.bumpStop.gap) - suspension.wheelTravel;
        const auto toDroop = suspension.wheelTravel - wheelTravelAtShaft(corner, -corner.droopStop.gap);

        std::printf("  %3s %8.1f %9.5f %8.2f %8.2f %8.4f %8.4f %8.2f %9.1f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n",
                    cornerAbbreviation(static_cast<Corner>(index)), solution.forces.tireVertical, q,
                    suspension.wheelTravel * 1000.0, shaft * 1000.0, spring->motionRatio, damper->motionRatio,
                    (corner.springFreeLength - spring->length) * 1000.0, solution.forces.spring, solution.forces.damper,
                    solution.forces.damper - viscous, solution.forces.bumpStop, solution.forces.droopStop,
                    solution.forces.antiRoll, toBump * 1000.0, toDroop * 1000.0);

        totalFz += solution.forces.tireVertical;
        (index < 2 ? frontFz : rearFz) += solution.forces.tireVertical;
    }

    const auto weight = state.chassis.mass * gravity;
    std::printf("  sum Fz %.1f N against weight %.1f N (%.4f), front axle %.1f N (%.2f%%), rear %.1f N\n", totalFz,
                weight, totalFz / weight, frontFz, 100.0 * frontFz / totalFz, rearFz);
    std::printf("  chassis mass %.2f kg, centre of mass (%.4f, %.4f, %.4f) body, body contacts %zu, kerb-path normal "
                "force FL %.1f N\n",
                state.chassis.mass, state.chassis.centreOfMass.x, state.chassis.centreOfMass.y,
                state.chassis.centreOfMass.z, step.contacts.points.size(), step.corners[0].obstacleNormalForce);
}

void printSetup(const VehicleSetup& setup)
{
    std::printf("\n==== the shipped setup ====\n");
    {
        const auto sprung = raceengine::computeMassProperties(setup.sprung);
        REQUIRE(sprung.has_value());
        std::printf("sprung mass %.2f kg at (%.4f, %.4f, %.4f), sprung inertia diag pitch(x) %.1f yaw(y) %.1f roll(z) %.1f kg.m2\n",
                    sprung->mass, sprung->centreOfMass.x, sprung->centreOfMass.y, sprung->centreOfMass.z,
                    sprung->inertia[0][0], sprung->inertia[1][1], sprung->inertia[2][2]);
    }
    std::printf("switches: geometricLoadPath %d drivelineReaction %d tyreThermal %d tyrePressure %d brakeThermal %d "
                "kerbContact %d frameAcceleration %d nonlinearGeometry %d kinematicReaction %d\n",
                setup.geometricLoadPath, setup.drivelineReaction, setup.tyreThermal, setup.tyrePressure,
                setup.brakeThermal, setup.kerbContact, setup.frameAcceleration, setup.nonlinearGeometry,
                setup.kinematicReaction);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        const auto spring = solveSpringKinematics(corner.hardpoints, springElementOf(corner.hardpoints), 0.0, 0.0);
        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        REQUIRE(spring.has_value());
        REQUIRE(design.has_value());
        const auto ratio = std::abs(spring->motionRatio);

        std::printf("  %s: springRate %.1f N/m shaft, MR(design) %.4f, wheel rate %.1f N/m, free length %.4f m, "
                    "unsprung %.1f kg, tyre rate %.0f N/m, tyre damping %.0f, friction %.1f N, ARB %.0f N/m\n",
                    cornerAbbreviation(static_cast<Corner>(index)), corner.springRate, spring->motionRatio,
                    corner.springRate * ratio * ratio, corner.springFreeLength, corner.unsprungMass,
                    corner.tireVerticalRate, corner.tireVerticalDamping, corner.damperFriction, corner.antiRollRate);
        std::printf("      damper curve (shaft m/s, N):");
        for (const auto& point : corner.damper.points)
        {
            std::printf(" (%.5f, %.1f)", point.x, point.y);
        }
        std::printf("\n      bump stop gap %.4f rate %.0f prog %.1f damping %.0f | droop gap %.4f rate %.0f prog %.1f "
                    "damping %.0f | range q [%.4f, %.4f]\n",
                    corner.bumpStop.gap, corner.bumpStop.rate, corner.bumpStop.progression, corner.bumpStop.damping,
                    corner.droopStop.gap, corner.droopStop.rate, corner.droopStop.progression,
                    corner.droopStop.damping, corner.hardpoints.droopAngle, corner.hardpoints.bumpAngle);
        std::printf("      wheel travel to bump-stop touch %.2f mm, to droop-stop touch %.2f mm, linkage limits %.2f / "
                    "%.2f mm, travelPerAngle %.4f m/rad, rollCentre %.4f m\n",
                    wheelTravelAtShaft(corner, corner.bumpStop.gap) * 1000.0,
                    wheelTravelAtShaft(corner, -corner.droopStop.gap) * 1000.0,
                    raceengine::solveCorner(corner.hardpoints, corner.hardpoints.bumpAngle, 0.0)->wheelTravel * 1000.0,
                    raceengine::solveCorner(corner.hardpoints, corner.hardpoints.droopAngle, 0.0)->wheelTravel * 1000.0,
                    design->travelPerAngle, [&] { auto s = design.value(); raceengine::computeRollCentre(corner.hardpoints, s); return s.rollCentreHeight; }());
    }
}

} // namespace

TEST_CASE("suspension characterisation: the shipped Golf, measured", "[.suspension-characterisation]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    // The car as it ships, with one control applied for the measurement: the cavity gas is held at
    // the ideal pressure it is quoted at (`tyrePressure` off is the documented inert configuration),
    // so the tyre's vertical rate is the stated 298926 N/m for the whole of every scenario rather
    // than wandering ~3% with the cavity temperature over the seconds a ring-down takes. Nothing else
    // is touched; `setup` is a copy and the production car is unchanged.
    auto characterised = built.value();
    characterised.tyrePressure = false;

    // The experimental arm, if one is named (`docs/rear-spring-sensitivity-brief.md`): the rear wheel
    // rate restated on this copy alone. Unset is whatever `golfGtiMk7()` states — 57000 when the study
    // ran, 28000 since 2026-09-08 later.
    const auto arm = rearWheelRateArm();
    if (arm > 0.0)
    {
        std::printf("\n==== experimental arm: rear wheel rate %.1f N/m, production untouched ====\n", arm);
        stateRearWheelRate(characterised, arm);
    }

    const auto droopArm = rearDroopArm();
    if (!droopArm.empty())
    {
        std::printf("\n==== experimental arm: rear droop stop '%s', production untouched ====\n", droopArm.c_str());
        stateRearDroopArm(characterised, droopArm);
    }

    if (frameAccelerationOff())
    {
        std::printf("\n==== control arm: frameAcceleration OFF, the corner equation before 2026-09-08 latest of all ====\n");
        characterised.frameAcceleration = false;
    }

    if (nonlinearGeometryOff())
    {
        std::printf("\n==== control arm: nonlinearGeometry OFF, the corner equation before 2026-09-08 latest of all (c) ====\n");
        characterised.nonlinearGeometry = false;
    }

    if (kinematicReactionOff())
    {
        std::printf("\n==== control arm: kinematicReaction OFF, the chassis reaction before 2026-09-08 latest of all (d) ====\n");
        characterised.kinematicReaction = false;
    }

    const auto& setup = characterised;

    const auto directory = outputDirectory();
    printSetup(setup);

    const auto flat = PhysicsWorld::create(generateProvingGround(plate(1600.0)).value());
    REQUIRE(flat.has_value());

    auto state = VehicleState{};

    // --- 1. static ---------------------------------------------------------------------------------
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        auto recorder = Recorder(directory + "/static.csv", setup);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, flat.value(), tick);
        REQUIRE(stepped.has_value());
        recorder.row(0.0, setup, state, stepped.value(), VehicleInput{}, flatRoad);
        printStatic("static, settled 6 s at rest, gas seeded at ideal", setup, state, stepped.value());
    }

    // --- 1b. the local wheel rate at static ride, measured: a small mass over one axle ---------------
    // Two masses small enough that the corner stays inside the linear region of the linkage and well
    // short of its stop at every arm (20 kg over the rear axle is 98 N per corner, 7 mm at 14 kN/m),
    // so the finite difference dFz / dtravel is the local rate at the static position and not a
    // secant across the stop approach the 50 kg sweep below measures.
    for (const auto front : {true, false})
    {
        auto recorder = Recorder(directory + (front ? "/wheelrate_front.csv" : "/wheelrate_rear.csv"), setup);
        std::printf("\n==== local wheel rate at static ride, %s axle (added mass at the axle station, settled 8 s) ====\n",
                    front ? "front" : "rear");

        auto restFz = std::array<double, cornerCount>{};
        auto restTravel = std::array<double, cornerCount>{};
        for (const auto added : {0.0, 10.0, 20.0})
        {
            auto loaded = setup;
            if (added > 0.0)
            {
                loaded.sprung.push_back(MassComponent{.mass = added,
                                                      .centre = glm::dvec3(0.0, 0.50, front ? frontAxle : rearAxle),
                                                      .inertia = glm::dmat3(0.0)});
            }

            settle(loaded, state, flat.value(), 0.0, 800.0, 2880);
            const auto stepped = stepVehicle(loaded, state, VehicleInput{}, noDriveTorque, flat.value(), tick);
            REQUIRE(stepped.has_value());
            recorder.row(added, loaded, state, stepped.value(), VehicleInput{}, flatRoad);

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto fz = stepped->corners[index].forces.tireVertical;
                const auto travel = stepped->corners[index].suspension.wheelTravel;
                if (added == 0.0)
                {
                    restFz[index] = fz;
                    restTravel[index] = travel;
                    continue;
                }

                const auto dTravel = travel - restTravel[index];
                std::printf("  +%.0f kg %s: dFz %+8.2f N, dtravel %+7.3f mm, local wheel rate %s\n", added,
                            cornerAbbreviation(static_cast<Corner>(index)), fz - restFz[index], dTravel * 1000.0,
                            std::abs(dTravel) > 1e-5 ? (std::to_string(std::lround((fz - restFz[index]) / dTravel)) + " N/m").c_str()
                                                     : "n/a (no travel)");
            }
        }
    }

    // --- 2. small-signal body modes ----------------------------------------------------------------
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        state.chassis.linearVelocity = glm::dvec3(0.0, 0.10, 0.0);
        auto recorder = Recorder(directory + "/heave.csv", setup);
        auto hold = SpeedHold{};
        run(setup, state, flat.value(), recorder, flatRoad, 2160, hold);
    }
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        setAngularVelocity(state.chassis, glm::dvec3(0.10, 0.0, 0.0));
        auto recorder = Recorder(directory + "/pitch.csv", setup);
        auto hold = SpeedHold{};
        run(setup, state, flat.value(), recorder, flatRoad, 2160, hold);
    }
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        setAngularVelocity(state.chassis, glm::dvec3(0.0, 0.0, 0.15));
        auto recorder = Recorder(directory + "/roll.csv", setup);
        auto hold = SpeedHold{};
        run(setup, state, flat.value(), recorder, flatRoad, 2160, hold);
    }

    // --- 2b. the same three at three times the amplitude, for the friction/viscous split -------------
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        state.chassis.linearVelocity = glm::dvec3(0.0, 0.30, 0.0);
        auto recorder = Recorder(directory + "/heave_large.csv", setup);
        auto hold = SpeedHold{};
        run(setup, state, flat.value(), recorder, flatRoad, 2160, hold);
    }
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        setAngularVelocity(state.chassis, glm::dvec3(0.30, 0.0, 0.0));
        auto recorder = Recorder(directory + "/pitch_large.csv", setup);
        auto hold = SpeedHold{};
        run(setup, state, flat.value(), recorder, flatRoad, 2160, hold);
    }
    {
        settle(setup, state, flat.value(), 0.0, 800.0);
        setAngularVelocity(state.chassis, glm::dvec3(0.0, 0.0, 0.45));
        auto recorder = Recorder(directory + "/roll_large.csv", setup);
        auto hold = SpeedHold{};
        run(setup, state, flat.value(), recorder, flatRoad, 2160, hold);
    }

    // --- 3. wheel hop, one corner per axle, at three impulse sizes -----------------------------------
    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        for (const auto rate : {0.3, 1.0, 3.0})
        {
            settle(setup, state, flat.value(), 0.0, 800.0);
            state.corners[index].wishboneRate = rate;
            const auto name = std::string(index == 0 ? "/hop_front" : "/hop_rear") + (rate < 0.5 ? "_small" : rate > 2.0 ? "_large" : "") + ".csv";
            auto recorder = Recorder(directory + name, setup);
            auto hold = SpeedHold{};
            run(setup, state, flat.value(), recorder, flatRoad, 1080, hold);
        }
    }

    // --- 4. single-wheel bumps and dips at 15 m/s ----------------------------------------------------
    const auto bumpStart = 40.0;
    const auto bumpLength = 2.0;
    for (const auto amplitude : {0.010, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, -0.010, -0.025, -0.040, -0.060})
    {
        const HeightFunction road = [=](const double x, const double z)
        {
            const auto onLeft = smoothstep(0.20, 0.45, x) * (1.0 - smoothstep(1.10, 1.35, x));
            return onLeft * versine(z, bumpStart, bumpLength, amplitude);
        };
        const auto mesh = heightfield(road, trackColumns, rows(-10.0, 90.0, bumpStart - 2.0, bumpStart + bumpLength + 2.0, 0.5, 0.05));
        const auto world = PhysicsWorld::create(mesh);
        REQUIRE(world.has_value());

        settle(setup, state, world.value(), 15.0, 0.0);
        auto name = std::string(amplitude < 0.0 ? "/dip_" : "/bump_") + std::to_string(static_cast<int>(std::lround(std::abs(amplitude) * 1000.0))) + ".csv";
        auto recorder = Recorder(directory + name, setup);
        auto hold = SpeedHold{.target = 15.0};
        run(setup, state, world.value(), recorder, road, 1800, hold);
    }

    // --- 4b. a kerb drop-off, 60 mm down over 0.3 m, the left track only, at 10 m/s ----------------
    {
        const HeightFunction road = [=](const double x, const double z)
        {
            const auto onLeft = smoothstep(0.20, 0.45, x) * (1.0 - smoothstep(1.10, 1.35, x));
            return onLeft * (-0.060 * smoothstep(bumpStart, bumpStart + 0.3, z));
        };
        const auto mesh = heightfield(road, trackColumns, rows(-10.0, 90.0, bumpStart - 2.0, bumpStart + 3.0, 0.5, 0.05));
        const auto world = PhysicsWorld::create(mesh);
        REQUIRE(world.has_value());

        settle(setup, state, world.value(), 10.0, 0.0);
        auto recorder = Recorder(directory + "/dropoff_60.csv", setup);
        auto hold = SpeedHold{.target = 10.0};
        run(setup, state, world.value(), recorder, road, 2160, hold);
    }

    // --- 5. deterministic rough road at 20 m/s, both tracks, stated sum of sines ----------------------
    {
        const auto roughStart = 30.0;
        const HeightFunction road = [=](const double x, const double z)
        {
            constexpr auto wavelengths = std::array{0.7, 1.3, 2.9, 5.3, 11.0};
            constexpr auto amplitudes = std::array{0.0015, 0.0025, 0.004, 0.006, 0.008};
            constexpr auto phasesLeft = std::array{0.0, 1.1, 2.3, 3.7, 5.1};
            constexpr auto phasesRight = std::array{2.0, 0.4, 4.1, 1.9, 3.3};

            const auto fade = smoothstep(roughStart, roughStart + 5.0, z);
            const auto onLeft = smoothstep(0.20, 0.45, x) * (1.0 - smoothstep(1.10, 1.35, x));
            const auto onRight = smoothstep(-1.35, -1.10, x) * (1.0 - smoothstep(-0.45, -0.20, x));

            auto left = 0.0;
            auto right = 0.0;
            for (auto index = std::size_t{0}; index < wavelengths.size(); index++)
            {
                left += amplitudes[index] * std::sin(2.0 * pi * z / wavelengths[index] + phasesLeft[index]);
                right += amplitudes[index] * std::sin(2.0 * pi * z / wavelengths[index] + phasesRight[index]);
            }

            return fade * (onLeft * left + onRight * right);
        };
        const auto mesh = heightfield(road, trackColumns, rows(-10.0, 260.0, roughStart - 2.0, 260.0, 0.5, 0.05));
        const auto world = PhysicsWorld::create(mesh);
        REQUIRE(world.has_value());

        settle(setup, state, world.value(), 20.0, 0.0);
        auto recorder = Recorder(directory + "/rough.csv", setup);
        auto hold = SpeedHold{.target = 20.0};
        run(setup, state, world.value(), recorder, road, 3960, hold);
    }

    // --- 6. a half-pedal stop from 25 m/s, nobody intervening --------------------------------------
    {
        settle(setup, state, flat.value(), 25.0, 800.0);
        auto recorder = Recorder(directory + "/brake_half.csv", setup);
        auto hold = SpeedHold{};
        run(setup, state, flat.value(), recorder, flatRoad, 1440, hold,
            [](const int step)
            {
                auto input = VehicleInput{};
                input.brake = step >= 180 ? 0.5 : 0.0;
                return input;
            });
    }

    // --- 7. steering steps at 20 m/s, speed held -----------------------------------------------------
    for (const auto steering : {0.15, 0.30})
    {
        settle(setup, state, flat.value(), 20.0, 800.0);
        auto name = std::string("/steer_") + std::to_string(static_cast<int>(std::lround(steering * 100.0))) + ".csv";
        auto recorder = Recorder(directory + name, setup);
        auto hold = SpeedHold{.target = 20.0};
        run(setup, state, flat.value(), recorder, flatRoad, 1800, hold,
            [steering](const int step)
            {
                auto input = VehicleInput{};
                input.steering = step >= 180 ? steering : 0.0;
                return input;
            });
    }

    // --- 8. quasi-static axle load sweep towards the bump stops ------------------------------------
    for (const auto front : {true, false})
    {
        auto recorder = Recorder(directory + (front ? "/load_front.csv" : "/load_rear.csv"), setup);
        std::printf("\n==== quasi-static load over the %s axle ====\n", front ? "front" : "rear");
        std::printf("  %7s %9s %9s %9s %9s %9s %9s %9s %9s\n", "kg", "Fz N", "trav mm", "shaft mm", "Fspr N",
                    "Fbump N", "Fdroop N", "y mm", "pitch deg");

        for (auto added = 0.0; added <= 900.0; added += 50.0)
        {
            auto loaded = setup;
            if (added > 0.0)
            {
                loaded.sprung.push_back(MassComponent{.mass = added,
                                                      .centre = glm::dvec3(0.0, 0.50, front ? frontAxle : rearAxle),
                                                      .inertia = glm::dmat3(0.0)});
            }

            settle(loaded, state, flat.value(), 0.0, 800.0, 2880);
            const auto stepped = stepVehicle(loaded, state, VehicleInput{}, noDriveTorque, flat.value(), tick);
            REQUIRE(stepped.has_value());
            recorder.row(added, loaded, state, stepped.value(), VehicleInput{}, flatRoad);

            const auto index = front ? std::size_t{0} : std::size_t{2};
            const auto& corner = setup.corners[index];
            const auto& solution = stepped->corners[index];
            const auto design = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).length;
            const auto shaft = design - raceengine::solveDamperGeometry(corner, solution.suspension).length;
            std::printf("  %7.0f %9.1f %9.2f %9.2f %9.1f %9.1f %9.1f %9.2f %9.3f\n", added, solution.forces.tireVertical,
                        solution.suspension.wheelTravel * 1000.0, shaft * 1000.0, solution.forces.spring,
                        solution.forces.bumpStop, solution.forces.droopStop, state.chassis.position.y * 1000.0,
                        stepped->telemetry.pitch * degrees);
        }
    }

    // --- 9. resting on the front bump stops: 400 kg over the front axle, the last 4 s of settle -----
    {
        auto loaded = setup;
        loaded.sprung.push_back(MassComponent{.mass = 400.0, .centre = glm::dvec3(0.0, 0.50, frontAxle), .inertia = glm::dmat3(0.0)});
        settle(loaded, state, flat.value(), 0.0, 800.0, 1440);
        auto recorder = Recorder(directory + "/stoprest_front.csv", loaded);
        auto hold = SpeedHold{};
        run(loaded, state, flat.value(), recorder, flatRoad, 1440, hold);
    }

    // --- 10. the skidpad criterion's own driven hold, on this car ---------------------------------
    // `GolfGtiTests.cpp`'s fixture carried across: 20 m/s held by the same PI drive on the front
    // wheels, ten seconds, the last second averaged, lateral acceleration and yaw rate stated toward
    // the car's own right. The angles are the criterion's own lattice across the gripping range and
    // the peak (0.30-0.45 on the shipped car), with the two lowest for the understeer gradient. Per
    // angle: the steady state, the front axle's share of lateral load transfer, the fixture's own
    // held-speed precondition (printed and written rather than asserted, because a probe reports and
    // does not abort), and every corner's stop participation over the whole hold. A positive demand
    // is a right turn, so the OUTSIDE wheels are the left pair, and those are the two stop columns.
    {
        std::printf("\n==== skidpad: the criterion's driven hold at 20 m/s, last second ====\n");
        std::printf("  %5s %7s %8s %6s %7s %7s | %6s %6s %6s %6s | %6s %6s %6s %6s | %5s %7s %5s %7s | %s\n", "steer",
                    "ay g", "yaw r/s", "v m/s", "roll", "fshare", "FzFL", "FzFR", "FzRL", "FzRR", "trFL", "trFR", "trRL",
                    "trRR", "stFL", "peak N", "stRL", "peak N", "held");

        auto* file = std::fopen((directory + "/skidpad.csv").c_str(), "w");
        REQUIRE(file != nullptr);
        std::fprintf(file, "steer,ay,yawRate,speed,roll,frontShare,slowest,fastest,worstKinematic,halfDifference,held");
        // The four columns after `lossTicks` were added for the 2026-09-08 post-adoption validation:
        // the tyre's own lateral and longitudinal force, its camber and its slip angle, each the
        // last-second mean, so the skidpad's balance can be read per tyre rather than per axle.
        for (const auto* name : {"fz", "trav", "fbTicks", "fbPeak", "fdrTicks", "lossTicks", "fy", "fx", "camber", "slipang"})
        {
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                std::fprintf(file, ",%s_%s", name, cornerAbbreviation(static_cast<Corner>(index)));
            }
        }
        std::fprintf(file, "\n");

        for (const auto steering : {0.06, 0.11, 0.15, 0.22, 0.30, 0.34, 0.36, 0.38, 0.40, 0.45, 0.60})
        {
            settle(setup, state, flat.value(), 20.0, 800.0);
            auto hold = SpeedHold{.target = 20.0};
            auto input = VehicleInput{};
            input.steering = steering;

            constexpr auto toTheRight = outboardSign(CornerSide::Right);
            auto lateral = 0.0;
            auto yaw = 0.0;
            auto speed = 0.0;
            auto roll = 0.0;
            auto samples = 0;
            auto slowest = 1e9;
            auto fastest = 0.0;
            auto worstKinematic = 0.0;
            auto firstHalf = 0.0;
            auto secondHalf = 0.0;
            auto fz = std::array<double, cornerCount>{};
            auto travel = std::array<double, cornerCount>{};
            auto bumpTicks = std::array<int, cornerCount>{};
            auto bumpPeak = std::array<double, cornerCount>{};
            auto droopTicks = std::array<int, cornerCount>{};
            auto lossTicks = std::array<int, cornerCount>{};
            auto lateralForce = std::array<double, cornerCount>{};
            auto longitudinalForce = std::array<double, cornerCount>{};
            auto camber = std::array<double, cornerCount>{};
            auto slipAngle = std::array<double, cornerCount>{};

            for (auto step = 0; step < 3600; step++)
            {
                const auto stepped = stepVehicle(setup, state, input, hold.torques(setup, state), flat.value(), tick);
                REQUIRE(stepped.has_value());

                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    const auto& corner = stepped->corners[index];
                    bumpTicks[index] += corner.forces.bumpStop > 0.0 ? 1 : 0;
                    bumpPeak[index] = std::max(bumpPeak[index], corner.forces.bumpStop);
                    droopTicks[index] += corner.forces.droopStop != 0.0 ? 1 : 0;
                    lossTicks[index] += corner.patch.inContact ? 0 : 1;
                }

                if (step < 3240)
                {
                    continue;
                }

                const auto ay = stepped->telemetry.acceleration.x * toTheRight;
                const auto yawRate = stepped->telemetry.yawRate * toTheRight;
                const auto carried = glm::length(state.chassis.linearVelocity);
                lateral += ay;
                yaw += yawRate;
                speed += carried;
                roll += stepped->telemetry.roll;
                samples++;
                slowest = std::min(slowest, carried);
                fastest = std::max(fastest, carried);
                worstKinematic = std::max(worstKinematic, std::abs(ay - carried * yawRate));
                (samples <= 180 ? firstHalf : secondHalf) += ay;
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    const auto& wheel = stepped->telemetry.wheels[index];
                    fz[index] += stepped->corners[index].forces.tireVertical;
                    travel[index] += stepped->corners[index].suspension.wheelTravel;
                    lateralForce[index] += wheel.forceLateral;
                    longitudinalForce[index] += wheel.forceLongitudinal;
                    camber[index] += wheel.camber;
                    slipAngle[index] += wheel.slipAngle;
                }
            }

            const auto n = static_cast<double>(samples);
            lateral /= n * gravity;
            yaw /= n;
            speed /= n;
            roll /= n;
            for (auto* values : {&fz, &travel, &lateralForce, &longitudinalForce, &camber, &slipAngle})
            {
                for (auto& value : *values)
                {
                    value /= n;
                }
            }
            const auto halfDifference = std::abs(firstHalf / 180.0 - secondHalf / (n - 180.0));
            const auto held = slowest > 0.9 * 20.0 && fastest < 1.1 * 20.0 &&
                              worstKinematic < 0.5 * std::abs(lateral) * gravity + 1.0 &&
                              halfDifference < 0.15 * std::abs(lateral) * gravity + 0.2;
            const auto frontTransfer = std::abs(fz[0] - fz[1]);
            const auto rearTransfer = std::abs(fz[2] - fz[3]);
            const auto frontShare = frontTransfer + rearTransfer > 1e-9 ? frontTransfer / (frontTransfer + rearTransfer) : 0.0;

            std::printf("  %5.2f %7.4f %8.4f %6.2f %7.3f %7.4f | %6.0f %6.0f %6.0f %6.0f | %6.1f %6.1f %6.1f %6.1f | %5d %7.0f %5d %7.0f | %s\n",
                        steering, lateral, yaw, speed, roll * degrees, frontShare, fz[0], fz[1], fz[2], fz[3],
                        travel[0] * 1000.0, travel[1] * 1000.0, travel[2] * 1000.0, travel[3] * 1000.0, bumpTicks[0],
                        bumpPeak[0], bumpTicks[2], bumpPeak[2], held ? "yes" : "NO");
            std::fflush(stdout);

            std::fprintf(file, "%.4f,%.6f,%.6f,%.4f,%.7f,%.6f,%.4f,%.4f,%.5f,%.5f,%d", steering, lateral, yaw, speed, roll,
                         frontShare, slowest, fastest, worstKinematic, halfDifference, held ? 1 : 0);
            for (const auto value : fz)
            {
                std::fprintf(file, ",%.3f", value);
            }
            for (const auto value : travel)
            {
                std::fprintf(file, ",%.6f", value);
            }
            for (const auto value : bumpTicks)
            {
                std::fprintf(file, ",%d", value);
            }
            for (const auto value : bumpPeak)
            {
                std::fprintf(file, ",%.1f", value);
            }
            for (const auto value : droopTicks)
            {
                std::fprintf(file, ",%d", value);
            }
            for (const auto value : lossTicks)
            {
                std::fprintf(file, ",%d", value);
            }
            for (const auto* values : {&lateralForce, &longitudinalForce})
            {
                for (const auto value : *values)
                {
                    std::fprintf(file, ",%.3f", value);
                }
            }
            for (const auto* values : {&camber, &slipAngle})
            {
                for (const auto value : *values)
                {
                    std::fprintf(file, ",%.6f", value);
                }
            }
            std::fprintf(file, "\n");
        }
        std::fclose(file);
    }

    // --- 13. the two driving criteria, replicated on the criteria's own plant -----------------------
    // `GolfGtiTests.cpp`'s criterion 6 (a 0.06 steering step at 25 m/s, coasting, three seconds) and
    // `TractionControlTests.cpp`'s 0-100 (its launch harness line for line: a second on the brake in
    // first, then full throttle with the harness's own road-speed upshift, traction control in Sport
    // for the criterion and off for the 6.842 s characterisation pin). Both run on a copy of the car
    // built exactly as those fixtures build it — cavity gas as shipped, their 1440-tick settle from
    // 0.572 m, their plates — with the experimental arm restated on it when one is named. Added for
    // the 2026-09-08 post-adoption validation: the knob-unset run reproduces the criteria's own
    // figures, and an arm gives the same criteria on the other car, which no criterion can do itself.
    {
        auto criterionCar = built.value();
        if (arm > 0.0)
        {
            stateRearWheelRate(criterionCar, arm);
        }
        if (!droopArm.empty())
        {
            stateRearDroopArm(criterionCar, droopArm);
        }
        if (frameAccelerationOff())
        {
            criterionCar.frameAcceleration = false;
        }
        if (nonlinearGeometryOff())
        {
            criterionCar.nonlinearGeometry = false;
        }
        if (kinematicReactionOff())
        {
            criterionCar.kinematicReaction = false;
        }

        constexpr auto criterionHeight = 0.572;
        constexpr auto criterionRadius = 0.3186;
        const auto criterionSettle = [&](VehicleState& car, const PhysicsWorld& world, const double speed)
        {
            car = VehicleState{};
            car.chassis.position = glm::dvec3(0.0, criterionHeight, 20.0);
            for (auto step = 0; step < 1440; step++)
            {
                REQUIRE(stepVehicle(criterionCar, car, VehicleInput{}, noDriveTorque, world, tick).has_value());
            }
            car.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
            for (auto& corner : car.corners)
            {
                corner.wheelSpeed = speed / criterionRadius;
            }
        };

        // Criterion 6.
        {
            const auto world = PhysicsWorld::create(generateProvingGround(plate(400.0)).value());
            REQUIRE(world.has_value());
            auto car = VehicleState{};
            criterionSettle(car, world.value(), 25.0);

            auto input = VehicleInput{};
            input.steering = 0.06;
            constexpr auto toTheRight = outboardSign(CornerSide::Right);

            auto* file = std::fopen((directory + "/stepsteer_06.csv").c_str(), "w");
            REQUIRE(file != nullptr);
            std::fprintf(file, "t,yawRate,ay,roll,rollRate");
            for (const auto* name : {"fz", "fy", "trav", "fb", "fdr", "farb", "camber", "slipang"})
            {
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%s_%s", name, cornerAbbreviation(static_cast<Corner>(index)));
                }
            }
            std::fprintf(file, "\n");

            auto history = std::vector<double>{};
            for (auto step = 0; step < 1080; step++)
            {
                const auto stepped = stepVehicle(criterionCar, car, input, noDriveTorque, world.value(), tick);
                REQUIRE(stepped.has_value());
                const auto yawRate = stepped->telemetry.yawRate * toTheRight;
                history.push_back(yawRate);

                std::fprintf(file, "%.6f,%.7f,%.6f,%.7f,%.7f", step * tick, yawRate,
                             stepped->telemetry.acceleration.x * toTheRight, stepped->telemetry.roll,
                             stepped->telemetry.rollRate);
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.3f", stepped->corners[index].forces.tireVertical);
                }
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.3f", stepped->telemetry.wheels[index].forceLateral);
                }
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.6f", stepped->corners[index].suspension.wheelTravel);
                }
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.3f", stepped->corners[index].forces.bumpStop);
                }
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.3f", stepped->corners[index].forces.droopStop);
                }
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.3f", stepped->corners[index].forces.antiRoll);
                }
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.6f", stepped->telemetry.wheels[index].camber);
                }
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::fprintf(file, ",%.6f", stepped->telemetry.wheels[index].slipAngle);
                }
                std::fprintf(file, "\n");
            }
            std::fclose(file);

            // The criterion's own arithmetic, unchanged.
            auto settled = 0.0;
            for (auto index = history.size() - 180; index < history.size(); index++)
            {
                settled += history[index];
            }
            settled /= 180.0;

            auto riseTicks = std::size_t{0};
            while (riseTicks < history.size() && history[riseTicks] < settled * 0.9)
            {
                riseTicks++;
            }

            const auto peak = *std::max_element(history.begin(), history.end());
            const auto tail = std::vector<double>(history.begin() + 540, history.end());
            const auto ripple = *std::max_element(tail.begin(), tail.end()) - *std::min_element(tail.begin(), tail.end());

            std::printf("\n==== criterion 6 replica: a 0.06 steering step at 25 m/s, coasting ====\n");
            std::printf("  settled yaw %.5f rad/s (bounds 0.08-0.26); rise to 90%% %.4f s (0.12-0.60); peak %.5f = %+.2f%% "
                        "overshoot (< 35%%); tail ripple %.6f = %.2f%% of settled (< 5%%)\n",
                        settled, static_cast<double>(riseTicks) * tick, peak, 100.0 * (peak / settled - 1.0), ripple,
                        100.0 * ripple / settled);
            std::fflush(stdout);
        }

        // The 0-100 harness.
        {
            auto descriptor = ProvingGroundDescriptor{};
            descriptor.length = 900.0;
            descriptor.width = 200.0;
            descriptor.cellSize = 2.0;
            descriptor.features = {};
            auto mesh = generateProvingGround(descriptor);
            REQUIRE(mesh.has_value());
            for (auto& material : mesh->materials)
            {
                material.gripMultiplier = 1.0;
                material.bumpiness = 0.0;
            }
            const auto world = PhysicsWorld::create(mesh.value());
            REQUIRE(world.has_value());

            const auto driveline = golfGtiMk7Driveline();
            const auto inertias = wheelInertias(criterionCar);
            constexpr auto hundred = 100.0 / 3.6;
            constexpr auto fifty = 50.0 / 3.6;
            constexpr auto noBrakePressure = std::array<double, cornerCount>{};

            for (const auto sport : {true, false})
            {
                auto assists = golfGtiMk7Assists(criterionCar);
                if (sport)
                {
                    assists.traction.mode = TractionMode::Sport;
                }

                auto car = VehicleState{};
                criterionSettle(car, world.value(), 0.0);

                auto drivelineState = DrivelineState{};
                startEngine(driveline, drivelineState);
                auto assistState = AssistState{};
                auto lastStep = VehicleStep{};
                auto road = std::array<double, cornerCount>{};

                const auto sense = [&]
                {
                    auto sensors = AssistSensors{};
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        sensors.wheelSpeeds[index] = car.corners[index].wheelSpeed;
                    }
                    sensors.yawRate = lastStep.telemetry.yawRate;
                    sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;
                    return sensors;
                };
                const auto speeds = [&]
                {
                    return std::array<double, cornerCount>{car.corners[0].wheelSpeed, car.corners[1].wheelSpeed,
                                                           car.corners[2].wheelSpeed, car.corners[3].wheelSpeed};
                };

                {
                    auto idling = VehicleInput{};
                    idling.brake = 1.0;
                    idling.gear = 1;
                    for (auto step = 0; step < 360; step++)
                    {
                        const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                                           brakeCircuitPressures(criterionCar, 1.0), tick);
                        const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
                        REQUIRE(torques.has_value());
                        const auto stepped = stepVehicle(criterionCar, car, idling, torques->wheel, world.value(), tick, command.brakes);
                        REQUIRE(stepped.has_value());
                        lastStep = stepped.value();
                        road = roadTorques(lastStep);
                    }
                }

                auto* file = std::fopen((directory + (sport ? "/launch_sport.csv" : "/launch_plain.csv")).c_str(), "w");
                REQUIRE(file != nullptr);
                std::fprintf(file, "t,speed,ax,pitch,rhF,rhR,gear,rpm,throttleScale,engineReduction");
                for (const auto* name : {"fz", "fx", "trav", "fb", "fdr", "slip", "tcBrake"})
                {
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%s_%s", name, cornerAbbreviation(static_cast<Corner>(index)));
                    }
                }
                std::fprintf(file, "\n");

                auto gear = 1;
                const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;
                auto toFifty = -1.0;
                auto toHundred = -1.0;
                auto peakSlip = 0.0;
                auto meanSlip = 0.0;
                auto peakReduction = 0.0;
                auto peakBrake = 0.0;
                auto firstBrake = -1.0;
                auto firstEngine = -1.0;
                auto samples = 0;
                auto stopTicks = std::array<int, cornerCount>{};
                auto stopPeak = std::array<double, cornerCount>{};
                auto droopTicks = std::array<int, cornerCount>{};
                auto travelMax = std::array<double, cornerCount>{};
                auto travelMin = std::array<double, cornerCount>{};
                auto fzMin = std::array<double, cornerCount>{1e9, 1e9, 1e9, 1e9};
                auto fzMax = std::array<double, cornerCount>{};
                auto pitchMax = 0.0;

                for (auto step = 1; step <= 20 * 360; step++)
                {
                    const auto now = static_cast<double>(step) * tick;
                    const auto roadSideSpeed =
                        std::abs(car.chassis.linearVelocity.z) / criterionRadius * driveline.gearbox.reduction(gear);
                    if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
                    {
                        gear++;
                    }

                    const auto command =
                        updateAssists(assists, assistState, sense(), {.brake = 0.0, .throttle = 1.0}, noBrakePressure, tick);
                    auto input = VehicleInput{};
                    input.throttle = command.throttleScale;
                    input.gear = gear;
                    const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
                    REQUIRE(torques.has_value());
                    const auto stepped = stepVehicle(criterionCar, car, input, torques->wheel, world.value(), tick, command.brakes);
                    REQUIRE(stepped.has_value());
                    lastStep = stepped.value();
                    road = roadTorques(lastStep);

                    peakReduction = std::max(peakReduction, command.channels.engineTorqueReduction);
                    for (const auto wheel : {frontLeft, frontRight})
                    {
                        peakBrake = std::max(peakBrake, command.channels.tractionBrakeTorque[wheel]);
                        if (firstBrake < 0.0 && command.channels.tractionBrakeTorque[wheel] > 1.0)
                        {
                            firstBrake = now;
                        }
                    }
                    if (firstEngine < 0.0 && command.channels.engineTorqueReduction > 0.05)
                    {
                        firstEngine = now;
                    }

                    const auto drivenSlip = 0.5 * (std::abs(lastStep.telemetry.wheels[frontLeft].slipRatio) +
                                                   std::abs(lastStep.telemetry.wheels[frontRight].slipRatio));
                    meanSlip += drivenSlip;
                    peakSlip = std::max(peakSlip, drivenSlip);
                    samples++;
                    pitchMax = std::max(pitchMax, std::abs(lastStep.telemetry.pitch));

                    std::fprintf(file, "%.6f,%.5f,%.5f,%.7f,%.5f,%.5f,%d,%.1f,%.4f,%.4f", now, car.chassis.linearVelocity.z,
                                 lastStep.telemetry.acceleration.z, lastStep.telemetry.pitch, lastStep.rideHeight.front,
                                 lastStep.rideHeight.rear, gear, drivelineState.engineSpeed, command.throttleScale,
                                 command.channels.engineTorqueReduction);
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        const auto& corner = lastStep.corners[index];
                        stopTicks[index] += corner.forces.bumpStop > 0.0 ? 1 : 0;
                        stopPeak[index] = std::max(stopPeak[index], corner.forces.bumpStop);
                        droopTicks[index] += corner.forces.droopStop != 0.0 ? 1 : 0;
                        travelMax[index] = std::max(travelMax[index], corner.suspension.wheelTravel);
                        travelMin[index] = std::min(travelMin[index], corner.suspension.wheelTravel);
                        fzMin[index] = std::min(fzMin[index], corner.forces.tireVertical);
                        fzMax[index] = std::max(fzMax[index], corner.forces.tireVertical);
                    }
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%.3f", lastStep.corners[index].forces.tireVertical);
                    }
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%.3f", lastStep.telemetry.wheels[index].forceLongitudinal);
                    }
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%.6f", lastStep.corners[index].suspension.wheelTravel);
                    }
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%.3f", lastStep.corners[index].forces.bumpStop);
                    }
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%.3f", lastStep.corners[index].forces.droopStop);
                    }
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%.5f", lastStep.telemetry.wheels[index].slipRatio);
                    }
                    for (auto index = std::size_t{0}; index < cornerCount; index++)
                    {
                        std::fprintf(file, ",%.3f", command.channels.tractionBrakeTorque[index]);
                    }
                    std::fprintf(file, "\n");

                    const auto speed = car.chassis.linearVelocity.z;
                    if (toFifty < 0.0 && speed >= fifty)
                    {
                        toFifty = now;
                    }
                    if (toHundred < 0.0 && speed >= hundred)
                    {
                        toHundred = now;
                        break;
                    }
                }
                std::fclose(file);

                meanSlip /= samples > 0 ? static_cast<double>(samples) : 1.0;
                std::printf("\n==== 0-100 replica, traction control %s ====\n", sport ? "SPORT (the criterion)" : "OFF (the 6.842 s pin)");
                std::printf("  0-50 %.4f s, 0-100 %.4f s; driven slip peak %.3f mean %.3f; engine reduction peak %.3f (first at %.3f s); "
                            "traction brake peak %.0f N.m (first at %.3f s); pitch peak %.3f deg\n",
                            toFifty, toHundred, peakSlip, meanSlip, peakReduction, firstEngine, peakBrake, firstBrake,
                            pitchMax * degrees);
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    std::printf("  %s: travel %+.2f / %+.2f mm, Fz %.0f / %.0f N, bump-stop ticks %d peak %.0f N, droop ticks %d\n",
                                cornerAbbreviation(static_cast<Corner>(index)), travelMax[index] * 1000.0,
                                travelMin[index] * 1000.0, fzMin[index], fzMax[index], stopTicks[index], stopPeak[index],
                                droopTicks[index]);
                }
                std::fflush(stdout);
            }
        }
    }

    std::printf("\nwrote CSVs to %s\n", directory.c_str());
}
