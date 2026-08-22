// RackTorque bodies. Declarations are in Input/Api/RackTorque.cppm.
//
// A **module implementation unit** — `module raceengine.input;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

module raceengine.input;

import :RackTorque;

namespace raceengine
{

namespace
{

[[nodiscard]] inline bool allFinite(const glm::dvec3& vector)
{
    return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

} // namespace

[[nodiscard]] RackTorque steeringRackTorque(const SteeringRack& rack, const std::span<const SteeredCorner> corners,
                                            const double rackVelocity, const double roadSpeed)
{
    auto result = RackTorque{};

    if (!std::isfinite(rackVelocity) || !std::isfinite(roadSpeed))
    {
        result.finite = false;

        return result;
    }

    for (auto index = std::size_t{0}; index < corners.size() && index < steeredCornerLimit; index++)
    {
        const auto& corner = corners[index];

        if (!allFinite(corner.lowerBallJoint) || !allFinite(corner.upperBallJoint) || !allFinite(corner.steeringArm) ||
            !allFinite(corner.rackOuter) || !allFinite(corner.contactPatch) || !allFinite(corner.patchNormal) ||
            !allFinite(corner.tyreForce) || !std::isfinite(corner.aligningMoment))
        {
            result.finite = false;

            return result;
        }

        const auto kingpinSpan = corner.upperBallJoint - corner.lowerBallJoint;
        if (glm::length(kingpinSpan) < 1e-9)
        {
            continue;
        }

        const auto kingpin = glm::normalize(kingpinSpan);
        const auto arm = corner.steeringArm - corner.lowerBallJoint;

        const auto moment =
            glm::dot(kingpin, glm::cross(corner.contactPatch - corner.lowerBallJoint, corner.tyreForce)) +
            corner.aligningMoment * glm::dot(kingpin, corner.patchNormal);

        result.kingpinTorque[index] = moment;

        const auto tieRod = corner.steeringArm - corner.rackOuter;
        if (glm::length(tieRod) < 1e-9)
        {
            continue;
        }

        const auto along = glm::normalize(tieRod);
        const auto aboutKingpin = glm::dot(kingpin, glm::cross(arm, along));

        // A steering arm sitting on the kingpin axis cannot be turned by the rack at all, which is
        // a legitimate corner — a rear axle with no steering authored is exactly this — and is a
        // zero to skip rather than a zero to divide by.
        if (std::abs(aboutKingpin) < 1e-9)
        {
            continue;
        }

        result.tyreForce += moment * along.x / aboutKingpin;
    }

    // Both resistances oppose the rack, so both take the sign of its motion and neither can do work
    // on it. Regularised rather than switched — see `frictionReferenceSpeed`.
    result.frictionForce = -rack.friction * std::tanh(rackVelocity / std::max(rack.frictionReferenceSpeed, 1e-9));
    result.dampingForce = -rack.damping * rackVelocity;
    result.rackForce = result.tyreForce + result.frictionForce + result.dampingForce;

    const auto radius = pinionRadius(rack);
    result.steeringTorque = result.rackForce * radius;

    // The motor's share, and it is solved rather than applied.
    //
    // A boost curve is drawn as assist against the *driver's* effort, and the driver's effort is
    // what is left after the assist — so writing it directly would be circular. What breaks the
    // circle is that the two add up: `rackForce = driver + assist`, and with the boost `b` defined
    // against the load the rack is under, `assist = b·driver` gives `driver = rackForce/(1 + b)`
    // in one line and no iteration.
    //
    // Then the cap, and it is the physical fact rather than a safety clamp: a motor has a peak
    // torque, and past it every further newton is the driver's. That is what leaves the incremental
    // gain at one where the cue matters most.
    const auto boost = assistBoost(rack.assist, result.rackForce, roadSpeed);
    const auto uncapped = result.rackForce * boost / (1.0 + boost);

    result.assistForce = std::copysign(std::min(std::abs(uncapped), std::max(rack.assist.maximumForce, 0.0)), uncapped);
    result.driverRackForce = result.rackForce - result.assistForce;
    result.assistedTorque = result.driverRackForce * radius;

    result.finite = std::isfinite(result.steeringTorque) && std::isfinite(result.assistedTorque);

    return result;
}

} // namespace raceengine
