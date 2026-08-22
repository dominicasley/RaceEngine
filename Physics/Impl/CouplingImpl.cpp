// Coupling bodies. Declarations are in Api/Coupling.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] CouplingSolution stepCoupling(const CouplingSetup& setup, CouplingState& state,
                                            const CouplingSides& sides, const double deltaTime)
{
    const auto drivingInertia = std::max(sides.drivingInertia, 1e-9);
    const auto drivenInertia = std::max(sides.drivenInertia, 1e-9);
    const auto capacity = std::max(sides.capacity, 0.0);
    const auto step = std::max(deltaTime, 1e-9);

    const auto slipSpeed = sides.drivingSpeed - sides.drivenSpeed;

    // Locked, the pair is one body: solve it as one and ask what torque the constraint had to carry
    // to make that true. Two terms and both are required. The first accelerates the two sides
    // together under the external torques; the second removes whatever slip is left, over one tick,
    // through the reduced admittance. A lock written with only the first holds whatever speed
    // difference it engaged at for ever and is not a lock.
    //
    // Each side is stated as what it takes to change its speed by one over a tick — `J/dt`, plus any
    // coefficient resisting that change. Written that way the driven side's spring joins as a term
    // rather than as a special case, and a side with no such coefficient gives back the plain
    // two-inertia formula this had before, exactly.
    const auto driving = drivingInertia / step;
    const auto driven = drivenInertia / step + std::max(sides.drivenResisting, 0.0);

    const auto total = driving + driven;
    const auto reduced = driving * driven / total;
    const auto required = (driven * sides.drivingTorque - driving * sides.drivenTorque) / total + reduced * slipSpeed;

    state.dwell += deltaTime;

    if (state.mode == CouplingMode::Locked)
    {
        if (std::abs(required) > capacity && state.dwell >= setup.lockDwell)
        {
            state.mode = CouplingMode::Slipping;
            state.dwell = 0.0;
        }
    }
    else if (std::abs(required) <= setup.lockFraction * capacity && std::abs(slipSpeed) <= setup.lockSlipSpeed &&
             state.dwell >= setup.slipDwell)
    {
        state.mode = CouplingMode::Locked;
        state.dwell = 0.0;
    }

    auto torque = 0.0;
    if (state.mode == CouplingMode::Locked || std::abs(slipSpeed) <= 1e-9)
    {
        // Held: it carries what the constraint asked for, clamped, because a coupling kept locked
        // by its dwell still cannot transmit more than it can hold — the dwell buys time against
        // hunting, not friction. Slipping at no slip is the same arithmetic for a different reason:
        // there is no sliding direction to read, and static friction takes what it is asked for up
        // to the same limit. Without that case a coupling caught at exactly zero would transmit its
        // whole capacity in an arbitrary direction.
        torque = std::clamp(required, -capacity, capacity);
    }
    else
    {
        // Sliding: it carries its capacity, signed by which way the surfaces are moving past each
        // other — but never more than `required`, which is the torque that brings the two sides
        // together exactly. A surface cannot reverse the slip it is damping, and over one tick a
        // capacity larger than the constraint does precisely that: it drives the driven side past
        // the driving one, the sliding direction flips, and the next tick flips it back.
        //
        // The cap only binds when the two agree in sign, because a capacity pushing slip *away*
        // from zero can never overshoot it. And it only binds at all when the driven side is light
        // enough for one tick's capacity to matter — a wheels-and-car driven side has a `required`
        // in the tens of thousands, so every rigid case reaches `sliding` unchanged.
        const auto sliding = std::copysign(capacity, slipSpeed);

        torque = sliding * required > 0.0 ? std::clamp(required, -capacity, capacity) : sliding;
    }

    state.torque = torque;

    return CouplingSolution{
        .torque = torque, .required = required, .slipSpeed = slipSpeed, .locked = state.mode == CouplingMode::Locked};
}

} // namespace raceengine
