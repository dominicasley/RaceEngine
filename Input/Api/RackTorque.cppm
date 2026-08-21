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

export module raceengine.input:RackTorque;

namespace raceengine
{

// Stage one of three: what the road is doing to the steering, in newton metres, with no notion
// anywhere in it of what is going to display that.
//
// It knows nothing about Fanatec, nothing about evdev, nothing about eight bits and nothing about
// eight newton metres. That is the whole point of it being its own partition. A road car's rack
// sees far more at the limit than any belt-driven base can produce, so something has to compress
// it — and the failure this separation exists to prevent is that the compression gets done by
// raising the tyre's self-aligning moment instead, at which point the *tyre model* is silently
// carrying a hardware compensation and the car feels right on one base and wrong everywhere else.
//
// Nothing here imports the physics module either, which is deliberate: what it takes is the
// geometry the kinematic solve already produced and the forces the tyre model already produced,
// as plain numbers, so the whole derivation can be pinned by tests that stand up no vehicle at
// all. The caller does the extraction, because the caller is the one holding a tick's result.
//
// Chassis frame throughout, SI: **+x is the car's left** (`outboardSign` in the physics module
// carries why, and the day it cost), +y up, +z forward, metres, newtons, radians.

// One steered corner, as the solve left it. Every point is where it *is* this tick, not where it
// was authored — which is what makes camber gain, bump steer and a wheel half on a kerb arrive
// through this path without a single term written for any of them.
export struct SteeredCorner
{
    // The kingpin axis, by its two ends. A double wishbone's upper end is its upper ball joint and
    // a strut's is its top bearing; the solve reports both in the same field, so nothing here has
    // to know which linkage it is holding.
    glm::dvec3 lowerBallJoint{0.0};
    glm::dvec3 upperBallJoint{0.0};

    // Where the tie rod picks up on the upright, solved, and where it picks up on the rack, with
    // the rack's travel already added. The line between them is the tie rod, and a tie rod is a
    // two-force member — which is what makes the whole steering Jacobian below closed form.
    glm::dvec3 steeringArm{0.0};
    glm::dvec3 rackOuter{0.0};

    glm::dvec3 contactPatch{0.0};
    glm::dvec3 patchNormal{0.0, 1.0, 0.0};

    // The tyre's whole resultant at the patch — vertical, longitudinal and lateral together — in
    // the chassis frame. All three matter and only one of them is obvious: the vertical load acting
    // through the kingpin's offset from the patch is what a parked car's steering weight *is*, and
    // it is also how a kerb reaches the driver's hands.
    glm::dvec3 tyreForce{0.0};

    // The self-aligning moment, N·m, as a couple about the patch normal. Pneumatic trail lives in
    // here; mechanical trail does not and must not, because mechanical trail is the geometry above
    // and would then be counted twice.
    double aligningMoment = 0.0;
};

// The steering box, stated in the units its parts are actually specified in.
export struct SteeringRack
{
    // Metres of rack per unit of steering demand, so full lock is this much travel. The vehicle
    // setup owns the number; it is copied here rather than derived, because this partition does not
    // import the one that states it.
    double travelPerInput = 0.055;

    // What the rim turns lock to lock. Together with the travel above this is the pinion radius,
    // and the pinion radius is the whole of the steering ratio on a rack and pinion: there is no
    // second reduction between the pinion and the rim, so torque at the pinion *is* torque at the
    // rim. That identity is why this stage can report newton metres at all.
    double lockToLockDegrees = 756.0;

    // Coulomb friction at the rack, newtons. Seal drag, the pinion's own mesh and the ball joints.
    // **Placeholder**, in the sense every number in this repository's vehicle data is: a documented
    // order of magnitude for a rack-and-pinion steering box rather than a measured figure for this
    // one. It is here rather than left out because a rack with no friction at all makes a wheel that
    // wanders on centre, which reads as a physics fault.
    double friction = 120.0;

    // Viscous damping at the rack, newtons per metre per second. Same provenance.
    double damping = 900.0;

    // Below this the Coulomb term is regularised rather than switched. A friction force written as
    // `-F * sign(v)` flips its whole magnitude between two consecutive ticks at a standstill, which
    // on a force feedback base is an audible buzz at the output rate and on a physics trace is a
    // square wave nobody can read past. `tanh` is the same force everywhere the rack is actually
    // moving and a steep ramp through zero where it is not.
    double frictionReferenceSpeed = 0.01;
};

// The pinion's radius, which is the rack's travel divided by the rim's rotation. Exported because
// it is the number that converts everything here between newtons and newton metres, and a reader
// checking the absolute magnitudes wants to see it.
export [[nodiscard]] inline double pinionRadius(const SteeringRack& rack)
{
    const auto lockToLock = rack.lockToLockDegrees * 0.017453292519943295;
    if (lockToLock <= 0.0)
    {
        return 0.0;
    }

    // Full travel is both ends: the demand runs -1 to 1.
    return 2.0 * rack.travelPerInput / lockToLock;
}

export inline constexpr std::size_t steeredCornerLimit = 2;

export struct RackTorque
{
    // Per corner, N·m about that corner's own kingpin axis. Diagnostic, and the first place to look
    // when the sign of the whole thing is wrong.
    std::array<double, steeredCornerLimit> kingpinTorque{};

    // Newtons at the rack, from the tyres alone, and then the two resistances that oppose the
    // rack's own motion. Split rather than summed because the tyre term is what the car is doing
    // and the other two are what the steering box is doing, and confusing the two is how a rack's
    // friction ends up being tuned to fix a tyre.
    double tyreForce = 0.0;
    double frictionForce = 0.0;
    double dampingForce = 0.0;
    double rackForce = 0.0;

    // **The deliverable.** Newton metres at the pinion, which on a rack and pinion is newton metres
    // at the rim. Positive in the same sense as a positive steering demand, so a self-aligning tyre
    // reports the opposite sign to the lock it is under.
    double steeringTorque = 0.0;

    // False when anything upstream handed this a value that is not a number. The caller must treat
    // that as a reason to let go of the wheel rather than as a number to pass on.
    bool finite = true;
};

namespace
{

[[nodiscard]] inline bool allFinite(const glm::dvec3& vector)
{
    return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

} // namespace

// The whole derivation, and it is two cross products and a ratio per corner.
//
// The tyre's resultant acts at the contact patch. Its moment about the kingpin axis is what the
// steering has to hold, and taking it as a moment about the *solved* axis is what makes mechanical
// trail, scrub radius, caster and kingpin inclination all arrive without any of them being written
// down: they are the geometry, and the geometry is an input.
//
//     T = k . [ (patch - lowerBallJoint) x F + Mz * n ]
//
// The tie rod then converts that to a force at the rack. It is a two-force member, so its force is
// along its own line; write u for that line, and the rack's own axis is chassis +x. A virtual
// displacement of the rack must keep the tie rod's length, which gives
//
//     dx * (u . x)  =  dtheta * ( k . ((arm - lowerBallJoint) x u) )
//
// so the ratio of those two brackets is dtheta/dx exactly, and virtual work turns the kingpin
// moment into a rack force with it. No small-angle approximation, no authored steering ratio and no
// lever arm measured off a drawing: change a hardpoint and this changes with it, which is the same
// rule the suspension solve keeps.
export [[nodiscard]] RackTorque
steeringRackTorque(const SteeringRack& rack, const std::span<const SteeredCorner> corners, const double rackVelocity)
{
    auto result = RackTorque{};

    if (!std::isfinite(rackVelocity))
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
    result.steeringTorque = result.rackForce * pinionRadius(rack);

    result.finite = std::isfinite(result.steeringTorque);

    return result;
}

// The stage-one trace, and it is the artefact rather than instrumentation.
//
// It is in newton metres at the rim whatever is plugged in, so a run on this base, a run on a
// direct drive base and a run with nothing attached at all produce the same numbers for the same
// lap. The stage-two columns ride beside it because a clip is only legible against what was asked
// for, but the stage-one column is the one that is comparable across hardware and sessions, and it
// is written whether or not any device took it.
export struct RackTorqueFrame
{
    double time = 0.0;
    std::uint64_t sequence = 0;

    // --- stage one, always, in N·m at the rim ---
    double steeringTorque = 0.0;
    double rackForce = 0.0;
    double tyreRackForce = 0.0;
    double rackTravel = 0.0;
    double rackVelocity = 0.0;

    // --- stage two, for this device ---
    double requestedTorque = 0.0;
    double commandedTorque = 0.0;
    double deliveredTorque = 0.0;
    bool clipped = false;

    // End to end, device report to the write returning, in milliseconds. Zero when nothing has been
    // written yet.
    double latencyMilliseconds = 0.0;
};

// The same ring the physics telemetry uses, and for the same reason: a validation run is minutes
// long and the interesting part is at the end, and a recorder that grew would allocate on the
// output thread.
export class RackTorqueRecorder
{
public:
    explicit RackTorqueRecorder(const std::size_t capacity) :
        frames(capacity)
    {
    }

    void record(const RackTorqueFrame& frame)
    {
        if (frames.empty())
        {
            return;
        }

        frames[next] = frame;
        next = (next + 1) % frames.size();
        filled = filled < frames.size() ? filled + 1 : frames.size();
    }

    // Oldest first, which is the opposite of storage order once the ring has wrapped.
    [[nodiscard]] std::vector<RackTorqueFrame> inOrder() const
    {
        auto ordered = std::vector<RackTorqueFrame>{};
        ordered.reserve(filled);

        const auto start = filled < frames.size() ? std::size_t{0} : next;
        for (auto offset = std::size_t{0}; offset < filled; offset++)
        {
            ordered.push_back(frames[(start + offset) % frames.size()]);
        }

        return ordered;
    }

    [[nodiscard]] std::size_t size() const
    {
        return filled;
    }

    [[nodiscard]] std::size_t capacity() const
    {
        return frames.size();
    }

private:
    std::vector<RackTorqueFrame> frames;
    std::size_t next = 0;
    std::size_t filled = 0;
};

namespace
{

inline void appendRackNumber(std::string& text, const double value, const int precision)
{
    auto buffer = std::array<char, 64>{};
    const auto written =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, precision);

    text.append(buffer.data(), written.ptr);
}

} // namespace

// Pure, like `telemetryToCsv` and for the same two reasons: what is in the text is worth testing and
// opening a file is not, and `std::to_chars` rather than a stream because a CSV written on a machine
// with a comma decimal separator is not a CSV.
export [[nodiscard]] inline std::string rackTorqueToCsv(const std::vector<RackTorqueFrame>& frames)
{
    auto text = std::string{};
    text.reserve(frames.size() * 128 + 512);

    text += "Time [s],Sequence,"
            "Steering Torque [Nm],Rack Force [N],Tyre Rack Force [N],Rack Travel [mm],Rack Vel [mm/s],"
            "Requested Torque [Nm],Commanded Torque [Nm],Delivered Torque [Nm],Clipped [],Latency [ms]\n";

    for (const auto& frame : frames)
    {
        appendRackNumber(text, frame.time, 6);
        text += ",";

        auto buffer = std::array<char, 32>{};
        const auto written = std::to_chars(buffer.data(), buffer.data() + buffer.size(), frame.sequence);
        text.append(buffer.data(), written.ptr);

        for (const auto value : {frame.steeringTorque, frame.rackForce, frame.tyreRackForce})
        {
            text += ",";
            appendRackNumber(text, value, 4);
        }

        for (const auto value : {frame.rackTravel * 1000.0, frame.rackVelocity * 1000.0})
        {
            text += ",";
            appendRackNumber(text, value, 3);
        }

        for (const auto value : {frame.requestedTorque, frame.commandedTorque, frame.deliveredTorque})
        {
            text += ",";
            appendRackNumber(text, value, 4);
        }

        text += frame.clipped ? ",1," : ",0,";
        appendRackNumber(text, frame.latencyMilliseconds, 3);
        text += "\n";
    }

    return text;
}

} // namespace raceengine
