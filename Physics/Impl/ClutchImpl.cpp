// Clutch bodies. Declarations are in Api/Clutch.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <string>
#include <type_traits>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] Curve roadClutchEngagement()
{
    return Curve{.points = {glm::dvec2(0.00, 1.00), glm::dvec2(0.35, 1.00), glm::dvec2(0.50, 0.62),
                            glm::dvec2(0.62, 0.22), glm::dvec2(0.75, 0.00), glm::dvec2(1.00, 0.00)}};
}

[[nodiscard]] double clutchCapacity(const FrictionClutch& clutch, const double pedal)
{
    const auto engagement = std::clamp(clutch.engagement.at(std::clamp(pedal, 0.0, 1.0)), 0.0, 1.0);

    return std::max(0.0, clutch.clampForce * clutch.frictionCoefficient * clutch.effectiveRadius *
                             static_cast<double>(clutch.faces) * engagement);
}

[[nodiscard]] double clutchPedalForCapacity(const FrictionClutch& clutch, const double capacity)
{
    const auto full =
        clutch.clampForce * clutch.frictionCoefficient * clutch.effectiveRadius * static_cast<double>(clutch.faces);
    const auto& points = clutch.engagement.points;

    if (points.empty() || !(full > 1e-9))
    {
        return 1.0;
    }

    const auto wanted = std::clamp(capacity, 0.0, full) / full;

    // Walked from the released end back, so a flat segment answers with its most-released pedal.
    for (auto index = points.size(); index-- > 1;)
    {
        const auto& clamped = points[index - 1];
        const auto& released = points[index];

        if (wanted <= clamped.y && wanted >= released.y)
        {
            const auto span = clamped.y - released.y;

            return span > 1e-12 ? clamped.x + (released.x - clamped.x) * (clamped.y - wanted) / span : released.x;
        }
    }

    // More clamp than the curve offers anywhere: the pedal is on its stop and the clutch gives what
    // it has.
    return points.front().x;
}

[[nodiscard]] Curve converterTorqueRatio()
{
    return Curve{.points = {glm::dvec2(0.00, 2.10), glm::dvec2(0.20, 1.85), glm::dvec2(0.40, 1.60),
                            glm::dvec2(0.60, 1.37), glm::dvec2(0.80, 1.13), glm::dvec2(0.88, 1.00),
                            glm::dvec2(1.00, 1.00)}};
}

[[nodiscard]] Curve converterCapacity()
{
    return Curve{.points = {glm::dvec2(0.00, 0.006308), glm::dvec2(0.20, 0.006046), glm::dvec2(0.40, 0.005495),
                            glm::dvec2(0.50, 0.005080), glm::dvec2(0.60, 0.004486), glm::dvec2(0.70, 0.003732),
                            glm::dvec2(0.80, 0.002751), glm::dvec2(0.90, 0.001470), glm::dvec2(1.00, 0.0)}};
}

[[nodiscard]] ConverterFlow converterFlow(const TorqueConverter& converter, const double impellerSpeed,
                                          const double turbineSpeed)
{
    // The speed ratio is always formed over the *faster* side, so there is never a small number
    // underneath it — an impeller at rest is only ill-conditioned if it is also the divisor. When
    // both sides are near rest the answer is a torque going as the square of a speed near zero,
    // which is nothing whatever ratio is read.
    const auto overrun = std::abs(turbineSpeed) > std::abs(impellerSpeed);
    const auto pumpSpeed = overrun ? turbineSpeed : impellerSpeed;

    if (std::abs(pumpSpeed) < 1e-9)
    {
        return ConverterFlow{};
    }

    // Clamped below zero as well as above, which is a decision rather than hygiene: a negative speed
    // ratio is the turbine turning against the impeller — a car rolling backwards in drive — and it
    // is read as stall. Published data there sits a little above the stall figure, so this is the
    // conservative side of it and the car still gets the full multiplication pushing it back up.
    const auto ratio = std::clamp((overrun ? impellerSpeed : turbineSpeed) / pumpSpeed, 0.0, 1.0);

    const auto scale = overrun ? converter.overrunCapacityScale : 1.0;
    const auto pumped = std::copysign(scale * converter.capacity.at(ratio) * pumpSpeed * pumpSpeed, pumpSpeed);

    if (!overrun)
    {
        return ConverterFlow{.impeller = pumped, .turbine = converter.torqueRatio.at(ratio) * pumped};
    }

    // Reverse flow: the turbine is driving and the stator's one-way clutch has let go, so there is
    // no reaction member and nothing to multiply against. Both sides see the same torque and it
    // opposes the turbine.
    return ConverterFlow{.impeller = -pumped, .turbine = -pumped};
}

[[nodiscard]] double advanceLockup(const ConverterLockup& lockup, const double apply, const double turbineSpeed,
                                   const std::int32_t gear, const double deltaTime)
{
    const auto bleed = std::max(apply - std::max(lockup.releaseRate * deltaTime, 0.0), 0.0);

    if (!lockup.enabled || gear < lockup.lowestGear)
    {
        return bleed;
    }

    if (turbineSpeed >= lockup.engageSpeed)
    {
        return std::min(apply + std::max(lockup.applyRate * deltaTime, 0.0), 1.0);
    }

    return turbineSpeed <= lockup.releaseSpeed ? bleed : apply;
}

[[nodiscard]] std::expected<DriveCouplingSolution, std::string>
stepDriveCoupling(const DriveCoupling& coupling, DriveCouplingState& state, const CouplingSides& sides,
                  const DriveCouplingCommand& command, const double deltaTime)
{
    switch (coupling.kind)
    {
    case DriveCouplingKind::FrictionClutch:
    {
        static_cast<void>(command.gear);

        auto loaded = sides;
        loaded.capacity = clutchCapacity(coupling.clutch, command.clutchPedal);

        const auto solution = stepCoupling(coupling.clutch.coupling, state.coupling, loaded, deltaTime);

        return DriveCouplingSolution{.drivingTorque = solution.torque,
                                     .drivenTorque = solution.torque,
                                     .slipSpeed = solution.slipSpeed,
                                     .slipPower = std::abs(solution.torque * solution.slipSpeed),
                                     .locked = solution.locked};
    }
    case DriveCouplingKind::TorqueConverter:
    {
        // A converter car has no third pedal, and this line is where that is said. Reading the field
        // and dropping it is the difference between a pedal that is ignored and a pedal nobody wired
        // up; fed into the fluid it would be a clutch that half works and reads as a feature.
        static_cast<void>(command.clutchPedal);

        const auto& converter = coupling.converter;
        const auto flow = converterFlow(converter, sides.drivingSpeed, sides.drivenSpeed);

        state.lockupApply =
            advanceLockup(converter.lockup, state.lockupApply, sides.drivenSpeed, command.gear, deltaTime);

        // The plate bridges the two sides, so what it has to hold is whatever the fluid has *not*
        // already taken out of them — hand it the raw external torques and it would be asked to
        // carry the fluid's work a second time.
        auto bridged = sides;
        bridged.drivingTorque = sides.drivingTorque - flow.impeller;
        bridged.drivenTorque = sides.drivenTorque + flow.turbine;
        bridged.capacity = std::max(converter.lockup.capacity, 0.0) * std::clamp(state.lockupApply, 0.0, 1.0);

        const auto plate = stepCoupling(converter.lockup.coupling, state.coupling, bridged, deltaTime);

        const auto slipSpeed = sides.drivingSpeed - sides.drivenSpeed;
        const auto fluidHeat = flow.impeller * sides.drivingSpeed - flow.turbine * sides.drivenSpeed;

        return DriveCouplingSolution{.drivingTorque = flow.impeller + plate.torque,
                                     .drivenTorque = flow.turbine + plate.torque,
                                     .slipSpeed = slipSpeed,
                                     .slipPower = std::max(fluidHeat, 0.0) + std::abs(plate.torque * slipSpeed),
                                     .locked = plate.locked};
    }
    }

    return std::unexpected("the drive coupling slot holds a kind this build has no model for");
}

void idleDriveCoupling(const DriveCoupling& coupling, DriveCouplingState& state, const double deltaTime)
{
    state.coupling = CouplingState{};

    switch (coupling.kind)
    {
    case DriveCouplingKind::FrictionClutch:
        state.lockupApply = 0.0;
        break;
    case DriveCouplingKind::TorqueConverter:
        state.lockupApply = advanceLockup(coupling.converter.lockup, state.lockupApply, 0.0, 0, deltaTime);
        break;
    }
}

[[nodiscard]] double advanceCreep(const AutoClutch& assist, const double command, const double brake,
                                  const double throttle, const bool inGear, const bool running, const double deltaTime)
{
    const auto asked = !assist.enabled || !assist.creep || !inGear || !running ||
                               std::clamp(throttle, 0.0, 1.0) > std::max(assist.creepThrottleLift, 0.0) ||
                               std::clamp(brake, 0.0, 1.0) >= std::max(assist.creepBrakeCut, 0.0)
                           ? 0.0
                           : std::max(assist.creepTorque, 0.0);

    const auto rate = std::max(asked > command ? assist.creepApplyRate : assist.creepReleaseRate, 0.0);
    const auto limit = rate * std::max(deltaTime, 0.0);

    return command + std::clamp(asked - command, -limit, limit);
}

[[nodiscard]] double creepPedal(const DriveCoupling& coupling, const double command)
{
    switch (coupling.kind)
    {
    case DriveCouplingKind::FrictionClutch:
        return clutchPedalForCapacity(coupling.clutch, command);
    case DriveCouplingKind::TorqueConverter:
        return 1.0;
    }

    return 1.0;
}

[[nodiscard]] double autoClutchPedal(const AutoClutch& assist, const double idleSpeed, const double clutchSideSpeed,
                                     const double engineSpeed, const double throttle, const bool inGear)
{
    if (!inGear)
    {
        return 1.0;
    }

    const auto release = assist.releaseFraction * idleSpeed;
    const auto grab = assist.grabFraction * idleSpeed;

    // The car has caught up with the gear it is in, so there is nothing left to slip for.
    //
    // **Signed, and it was `std::abs`.** The sign of the gear is already in this number — a reverse
    // ratio is negative, so a car reversing in reverse arrives positive exactly as a car driving
    // forward in first does — so taking the magnitude said something quite different from the sentence
    // above: that a car rolling *backwards* in a forward gear had caught up with it, and grew the
    // clamp with how fast it was rolling away downhill.
    //
    // **It was held back for a while and then became free, which is worth recording.** Measured
    // against the launch regulator this file used to carry, the correction moved the driving parity
    // gate by 23,191 of 32,400 blocks — because that regulator held the clutch open on the sandbox's
    // cold start, let it slam 480 N.m into a stationary tyre, and rang the compliant shaft backwards
    // through zero for 25 of the first 130 ticks, as far as −88.3 rad/s. `std::abs` read that as 0.74
    // of the way to fully clamped. With the regulator gone the shaft never goes backwards there at all
    // (`[.creep-coldstart]` measures 0.0000 rad/s) and this correction is **byte-identical on both
    // gates**. Two changes that each looked like they moved the frame, and only one of them did.
    const auto caught = std::clamp((clutchSideSpeed - release) / std::max(grab - release, 1e-9), 0.0, 1.0);

    // And the driver's request, which closes the clutch **directly** rather than through the engine's
    // speed.
    //
    // What was here was a launch regulator: engagement scaled by how far the engine had climbed above
    // idle toward `launchSpeed`, so at full throttle from rest the clutch stayed open, the engine
    // flared to two and a half thousand, and the clamp only came in against those revs. It reads from
    // the seat as a clutch that will not engage — the engine screaming and the car going nowhere —
    // and no dual clutch behaves that way. A DSG closes the clutch when you ask it for torque.
    //
    // So the request is the pedal, and nothing else. The engine's speed does not appear: it is the
    // anti-stall's business alone, which is the one rule that is allowed to open a clutch the driver
    // has asked to be shut.
    const auto asked = std::clamp(std::clamp(throttle, 0.0, 1.0) / std::max(assist.engageThrottle, 1e-9), 0.0, 1.0);

    static_cast<void>(engineSpeed);

    return 1.0 - std::max(caught, asked);
}

[[nodiscard]] double antiStallPedal(const AutoClutch& assist, const double idleSpeed, const double clutchSideSpeed,
                                    const double engineSpeed)
{
    if (!assist.enabled || clutchSideSpeed > assist.grabFraction * idleSpeed)
    {
        return 0.0;
    }

    const auto begin = assist.antiStallBegin * idleSpeed;
    const auto open = assist.antiStallOpen * idleSpeed;

    return std::clamp((begin - engineSpeed) / std::max(begin - open, 1e-9), 0.0, 1.0);
}

[[nodiscard]] double advanceClutchPedal(const AutoClutch& assist, const double pedal, const double driverPedal,
                                        const double automatic, const double antiStall, const double deltaTime)
{
    const auto driver = std::clamp(driverPedal, 0.0, 1.0);

    // No rate limit on the driver: a real pedal answers an ankle directly, and dumping it is a thing
    // a driver is allowed to do and this model is supposed to have an opinion about. The driver's
    // foot beats the anti-stall too — a human deliberately biting the clutch is re-specifying what
    // the engine is for, exactly as the cruise-control rule above says.
    if (!assist.enabled || driver > assist.freePlay)
    {
        return driver;
    }

    // Because the state above has been tracking the driver's pedal all the while it was the
    // driver's, the automation picks up where the foot left it instead of stepping.
    const auto limit = std::max(assist.pedalRate * deltaTime, 0.0);
    const auto followed =
        std::clamp(pedal + std::clamp(std::clamp(automatic, 0.0, 1.0) - pedal, -limit, limit), 0.0, 1.0);

    // The anti-stall is a floor past the rate limit, deliberately: the limit models a foot, and on
    // the hardware this automation stands in for — a dual clutch's hydraulics — the emergency
    // disengage is not a foot. Rate-limited, it loses the race it exists for: a full brake
    // application locks the wheels in a few tens of milliseconds and the pedal needs a quarter of a
    // second, so the engine is below its floor before the follow arrives. The floored value becomes
    // the state, so recovery decays smoothly from wherever the emergency put it.
    return std::max(followed, std::clamp(antiStall, 0.0, 1.0));
}

} // namespace raceengine
