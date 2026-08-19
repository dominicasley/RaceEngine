module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

export module raceengine.physics:Coupling;

namespace raceengine
{

// A friction coupling between two rotating inertias: it either holds them together or lets them
// slide, and the interesting part is entirely in when it changes its mind.
//
// Written once and owned by nothing. The friction clutch, the torque converter's lockup clutch and
// the differential's pack are the same device with a different capacity in front of it, and three
// copies of this state machine are three places for the hysteresis to be got subtly wrong. Nothing
// in here names a clutch, a converter or a differential for that reason.
//
// It integrates no speed. The coupling reports the torque it is carrying and the caller puts it on
// both sides, which is the rule the rest of the vehicle model works under.

export enum class CouplingMode : std::uint32_t { Slipping, Locked };

export struct CouplingSetup
{
    // Two thresholds rather than one, and that is not a refinement. With a single threshold the
    // tick that locks removes the very slip that justified locking, so the next tick unlocks: the
    // torque trace buzzes at the timestep's frequency, which is what hunting looks like in
    // telemetry. Locking is therefore considered at a fraction of what it takes to break a lock.
    double lockFraction = 0.8;
    // And it will not lock across a speed difference it would have to slam shut on.
    double lockSlipSpeed = 1.0;

    // A second threshold is still not enough on its own: a torque that dithers across *both* of
    // them hunts just as happily as one that dithers across a single one. What stops that is
    // refusing to answer for a while, which is what these are. In each direction, because a
    // coupling that may unlock instantly after locking is a coupling with no lock dwell at all.
    double slipDwell = 0.02;
    double lockDwell = 0.02;
};

export struct CouplingState
{
    CouplingMode mode = CouplingMode::Slipping;
    // Time in the current mode, reset on every transition. The dwells above are read against it.
    double dwell = 0.0;
    // What it carried last tick, kept because that is the trace a buzz shows up in.
    double torque = 0.0;
};

static_assert(std::is_trivially_copyable_v<CouplingState>, "a coupling's state is saved and restored by copying it");
static_assert(std::is_standard_layout_v<CouplingState>, "and rollback will later");

// The two sides as the coupling needs to see them. `capacity` is what the friction surface will
// hold before it gives, and is the one number that says which device this is standing in for.
export struct CouplingSides
{
    double drivingSpeed = 0.0;
    double drivenSpeed = 0.0;
    double drivingInertia = 1.0;
    double drivenInertia = 1.0;
    double drivingTorque = 0.0;
    double drivenTorque = 0.0;
    double capacity = 0.0;
};

export struct CouplingSolution
{
    // Positive drives the driven side forward, and is reacted on the driving side.
    double torque = 0.0;
    // What holding the two together would have cost this tick, whether or not it was held. This is
    // the number the capacity is judged against, and it is reported rather than left to be
    // recomputed by whoever wants to know why the lock broke.
    double required = 0.0;
    double slipSpeed = 0.0;
    bool locked = false;
};

// One tick of a coupling. Pure in (setup, sides, dt) apart from the mode and dwell it carries.
export [[nodiscard]] CouplingSolution stepCoupling(const CouplingSetup& setup, CouplingState& state,
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
    // through the reduced inertia. A lock written with only the first holds whatever speed
    // difference it engaged at for ever and is not a lock.
    const auto total = drivingInertia + drivenInertia;
    const auto reduced = drivingInertia * drivenInertia / total;
    const auto required = (drivenInertia * sides.drivingTorque - drivingInertia * sides.drivenTorque) / total +
                          reduced * slipSpeed / step;

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
        // other and by nothing else.
        torque = std::copysign(capacity, slipSpeed);
    }

    state.torque = torque;

    return CouplingSolution{
        .torque = torque, .required = required, .slipSpeed = slipSpeed, .locked = state.mode == CouplingMode::Locked};
}

} // namespace raceengine
