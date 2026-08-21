module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <type_traits>

#include <glm/glm.hpp>

export module raceengine.physics:Clutch;

import :Coupling;
import :Vehicle;

namespace raceengine
{

// The element between the engine's inertia and the gearbox input. It is an *element* rather than a
// clutch because a torque converter sits in the same slot, and nothing downstream may know which is
// fitted: the driveline asks this what the engine felt and what the gearbox got, which is the same
// sentence either way.
//
// Two kinds behind one function and one output struct, which is `SuspensionKind`'s shape and
// deliberately not an abstract base's. What a driveline wants from this slot does not vary by type;
// the pedal a friction clutch reads and the gear a converter's lockup reads are both in the
// arguments, and each kind discards the other's. Value semantics throughout, so `DrivelineSetup`
// stays assignable.
export enum class DriveCouplingKind : std::uint32_t { FrictionClutch, TorqueConverter };

// Pedal travel against how much of the clamp load reaches the plate, and it is data rather than code
// because a real clutch is nowhere near linear in it. The top of the travel is free play in the
// release mechanism, almost the whole of the clamp goes in over a narrow band in the middle, and the
// bottom is a plate already fully released. That band is what a driver launches on, and its width is
// the difference between a pedal and a switch.
export [[nodiscard]] Curve roadClutchEngagement()
{
    return Curve{.points = {glm::dvec2(0.00, 1.00), glm::dvec2(0.35, 1.00), glm::dvec2(0.50, 0.62),
                            glm::dvec2(0.62, 0.22), glm::dvec2(0.75, 0.00), glm::dvec2(1.00, 0.00)}};
}

export struct FrictionClutch
{
    // Placeholder, and a single 240 mm organic plate's numbers: 8 kN of diaphragm clamp, 0.30 dry
    // friction, a 0.100 m mean effective radius and the two faces one plate has come to 480 N.m,
    // which is 1.37 times what this engine makes. Sized under that it slips in top gear on its own.
    double clampForce = 8000.0;
    double frictionCoefficient = 0.30;
    double effectiveRadius = 0.100;
    std::uint32_t faces = 2;

    Curve engagement = roadClutchEngagement();

    CouplingSetup coupling;
};

// Torque capacity at a given pedal. Clamp force through the friction surfaces, scaled by where the
// pedal has the clamp — nothing here integrates or remembers anything.
export [[nodiscard]] double clutchCapacity(const FrictionClutch& clutch, const double pedal)
{
    const auto engagement = std::clamp(clutch.engagement.at(std::clamp(pedal, 0.0, 1.0)), 0.0, 1.0);

    return std::max(0.0, clutch.clampForce * clutch.frictionCoefficient * clutch.effectiveRadius *
                             static_cast<double>(clutch.faces) * engagement);
}

// The other kind, and it is not a variant of the friction clutch. A converter is a fluid coupling
// with a third member, and all it shares with a plate is where it sits in the chain: the impeller
// throws oil at the turbine, the stator turns what comes back and reacts the difference into the
// housing, and that reaction is the extra torque. Nothing in here has a capacity, a pedal or a
// sliding surface.
//
// **Both curves below are representative rather than measured**, which is a distinction nothing else
// in this model has to make and is therefore worth making loudly. Every other number here comes out
// of a real car's own data, and neither car on this machine is a torque-converter automatic — the
// Golf is a dual clutch and the M3 is a manual — so there is nothing to extract these from. The
// shape and the sizing are the published single-stage three-element characteristic for a mid-size
// passenger car: Wong, *Theory of Ground Vehicles* 4th ed. ch. 3; Kotwicki, "Dynamic Models for
// Torque Converter Equipped Vehicles", SAE 820393; Heisler, *Advanced Vehicle Technology* 2nd ed.
// ch. 3.

// Speed ratio against torque ratio. Just over two at stall, falling to one at the coupling point
// near 0.88 and staying there — past that the stator freewheels and the converter is a plain fluid
// coupling. Efficiency is the product of the two, and this shape reaches 0.90 by a speed ratio of
// 0.8 — the published figure. Past that it keeps creeping toward one rather than turning over as a
// measured curve does: slip is the only loss modelled here, and the churning and pumping losses that
// bend a real curve back down would be a third table. There is next to no torque left up there to be
// inefficient with, which is why the omission costs nothing and why a lockup clutch exists anyway.
export [[nodiscard]] Curve converterTorqueRatio()
{
    return Curve{.points = {glm::dvec2(0.00, 2.10), glm::dvec2(0.20, 1.85), glm::dvec2(0.40, 1.60),
                            glm::dvec2(0.60, 1.37), glm::dvec2(0.80, 1.13), glm::dvec2(0.88, 1.00),
                            glm::dvec2(1.00, 1.00)}};
}

// Speed ratio against the capacity coefficient C = T_impeller / omega_impeller^2, in N.m per
// (rad/s)^2.
//
// The literature quotes the *capacity factor* K = omega_impeller / sqrt(T_impeller) instead, and
// this is that table squared and inverted: 140, 143, 150, 156, 166, 182, 212, 290 and infinity
// rpm/sqrt(lb.ft) at the speed ratios below, which is an ordinary passenger-car converter. Held as C
// rather than as K because K is *infinite* where the two wheels come into step. Interpolating a
// piecewise-linear K toward a large finite stand-in instead collapses the torque to nothing well
// before the speed ratio gets there — which is precisely the range a converter without a lockup
// spends its life cruising in. C is zero there, which is both finite and the truth: no relative
// motion, no flow, no torque.
export [[nodiscard]] Curve converterCapacity()
{
    return Curve{.points = {glm::dvec2(0.00, 0.006308), glm::dvec2(0.20, 0.006046), glm::dvec2(0.40, 0.005495),
                            glm::dvec2(0.50, 0.005080), glm::dvec2(0.60, 0.004486), glm::dvec2(0.70, 0.003732),
                            glm::dvec2(0.80, 0.002751), glm::dvec2(0.90, 0.001470), glm::dvec2(1.00, 0.0)}};
}

// The plate that bridges impeller and turbine once there is nothing left to multiply. It is in
// *parallel* with the fluid rather than downstream of it, which is what a real lockup clutch is
// bolted across, and it is `stepCoupling`'s third consumer for the reason that partition names no
// device: a lockup plate is a friction clutch with a different capacity in front of it.
export struct ConverterLockup
{
    bool enabled = true;

    // What the plate holds fully applied. 1.3 times what this engine makes, sized as the friction
    // clutch's is and for the same reason.
    double capacity = 450.0;

    // Turbine speed, rad/s. Two of them, and the gap is not a refinement: a plate answering one
    // threshold chatters every time the speed sits on it. Between them the apply level is simply
    // held, which makes that level its own latch and is why nothing else has to carry one.
    double releaseSpeed = 140.0;
    double engageSpeed = 180.0;
    // A lockup in first is a car with no launch device at all.
    std::int32_t lowestGear = 3;

    // Fractions of full apply per second. A real plate is ramped by the transmission controller over
    // about half a second, and it is that ramp rather than any threshold that stops the engagement
    // being a bang.
    double applyRate = 2.5;
    double releaseRate = 5.0;

    // It will lock across far more slip than the gearbox plate will, and that is not a relaxation of
    // the same rule. A friction clutch is closed by a driver who has already matched the two speeds,
    // so a rad/s is all it should ever have to slam shut on; a lockup plate is closed against
    // whatever slip the fluid has left it, which at a converter's coupling point is tens of rpm and
    // never less. Held at a rad/s the plate simply never engages, which reads as a lockup that was
    // never wired up.
    CouplingSetup coupling{.lockSlipSpeed = 12.0};
};

export struct TorqueConverter
{
    Curve torqueRatio = converterTorqueRatio();
    Curve capacity = converterCapacity();

    // Reverse flow, where the turbine is the pump. Representative, like the curves: published
    // reverse-region capacity is well under the forward figure, and this is why an automatic gives
    // so little engine braking.
    double overrunCapacityScale = 0.5;

    ConverterLockup lockup;
};

// What the fluid is doing. `impeller` is what the converter takes off the engine and `turbine` what
// it delivers to the gearbox input; they are not the same number, and the difference is what the
// stator reacts into the housing.
export struct ConverterFlow
{
    double impeller = 0.0;
    double turbine = 0.0;
};

// The fluid path alone, with no lockup in it. Pure, so what a converter does can be swept and read
// off without a car around it.
export [[nodiscard]] ConverterFlow converterFlow(const TorqueConverter& converter, const double impellerSpeed,
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

// Where the lockup's apply pressure goes next, given what the car is doing.
export [[nodiscard]] double advanceLockup(const ConverterLockup& lockup, const double apply, const double turbineSpeed,
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

export struct DriveCoupling
{
    DriveCouplingKind kind = DriveCouplingKind::FrictionClutch;

    FrictionClutch clutch;
    TorqueConverter converter;
};

// The slot's own state. `CouplingState` is the lock/slip machine — the plate's for a clutch, the
// lockup's for a converter — and the apply level beside it is the one thing a converter integrates
// that a plate does not.
export struct DriveCouplingState
{
    CouplingState coupling{};
    double lockupApply = 0.0;
};

static_assert(std::is_trivially_copyable_v<DriveCouplingState>, "the driveline's state is saved by copying its bytes");
static_assert(std::is_standard_layout_v<DriveCouplingState>, "and rollback will later");

// What the slot is told each tick. Both fields are read by one kind and deliberately discarded by
// the other, which is exactly why they arrive here rather than in either device's own setup: what
// the driveline knows is that there is an element in the chain, and it may not know which.
export struct DriveCouplingCommand
{
    // 0 released and 1 fully depressed.
    double clutchPedal = 0.0;
    std::int32_t gear = 0;
};

export struct DriveCouplingSolution
{
    // What the slot took off the engine, and what it delivered to the gearbox input. One number does
    // for a friction clutch, whose two faces carry the same torque; a converter has a third member
    // grounded to the housing and the difference between these two is precisely what it reacts. A
    // slot reporting a single torque is a slot no converter can be honest in.
    double drivingTorque = 0.0;
    double drivenTorque = 0.0;

    double slipSpeed = 0.0;

    // Watts, and the caller integrates it. Each kind states its own loss because they are not the
    // same arithmetic: a plate dissipates torque times slip, a converter dissipates the power in
    // less the power out, and the two agree only where the torque ratio is one.
    double slipPower = 0.0;

    bool locked = false;
};

// One tick of whatever is fitted.
export [[nodiscard]] std::expected<DriveCouplingSolution, std::string>
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

// What the slot does on a tick where nothing is connected through it — neutral, a gear change, or a
// car with no driven axle. It is here rather than in the driveline because the two kinds answer
// differently and the driveline may not know which is fitted.
//
// A plate simply forgets which side of its hysteresis it was on: there is nothing to hold. A
// converter's lockup *bleeds* rather than dropping, because the driveline used to clear this whole
// struct and that put a step in the one channel a shift is judged by — a plate that vanished in a
// tick and took half a second to come back, on every gear change, for no reason a transmission
// controller would recognise.
export void idleDriveCoupling(const DriveCoupling& coupling, DriveCouplingState& state, const double deltaTime)
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

// A layer over the *pedal*, not around the clutch. It produces a pedal position and nothing else, so
// the friction model underneath is identical whoever is pressing — which is the whole point, because
// the rig has a real clutch pedal and a human slipping it from rest is the best test this model has.
export struct AutoClutch
{
    bool enabled = true;

    // Where the clutch must be fully out and fully in, measured on the driveline side and stated as
    // fractions of idle speed so that they follow the engine rather than being restated against it.
    double releaseFraction = 0.40;
    double grabFraction = 1.20;

    // Where it holds the engine on a launch, at full throttle.
    double launchSpeed = 250.0;

    // A foot has a speed limit and so does this. Without it the pedal is a step input into a
    // friction element, which is the one thing a coupling cannot be asked to answer sensibly.
    double pedalRate = 4.0;

    // The free play a real pedal already has at the top of its travel, and the line this uses to
    // decide that the driver is on the pedal.
    double freePlay = 0.05;

    // The anti-stall band, as fractions of idle speed. Above `antiStallBegin` the engine is healthy
    // and this contributes nothing; by `antiStallOpen` the clutch is fully out. It begins *below*
    // idle so the governor's own excursions never brush it, and it is fully open well above the
    // engine's stall floor — 0.55 of an 850 rpm idle is 468 rpm against a stall floor of 382 —
    // because the whole term exists to keep the fight away from that floor, not to referee at it.
    double antiStallBegin = 0.90;
    double antiStallOpen = 0.55;
};

// The pedal an automatic foot would be holding, given what the car is doing. `idleSpeed` arrives as a
// number rather than as the engine model because `:Driveline` imports this partition and not the
// other way about.
export [[nodiscard]] double autoClutchPedal(const AutoClutch& assist, const double idleSpeed,
                                            const double clutchSideSpeed, const double engineSpeed,
                                            const double throttle, const bool inGear)
{
    if (!inGear)
    {
        return 1.0;
    }

    const auto release = assist.releaseFraction * idleSpeed;
    const auto grab = assist.grabFraction * idleSpeed;

    // The car has caught up with the gear it is in, so there is nothing left to slip for.
    const auto caught = std::clamp((std::abs(clutchSideSpeed) - release) / std::max(grab - release, 1e-9), 0.0, 1.0);

    // And the launch. Revs the driver has asked for are revs this is allowed to spend, so the
    // engagement closes as the engine rises and opens again the moment the clutch drags it down.
    // That feedback is what holds a launch near `launchSpeed` without anything here naming a slip
    // target or reading the capacity it is about to command.
    const auto headroom =
        std::clamp((engineSpeed - idleSpeed) / std::max(assist.launchSpeed - idleSpeed, 1e-9), 0.0, 1.0);

    return 1.0 - std::max(caught, headroom * std::clamp(throttle, 0.0, 1.0));
}

// How far the clutch must be out *right now* because the engine is being dragged toward its stall
// floor. Zero for a healthy engine, one by `antiStallOpen`, and computed from the engine's own speed
// rather than the driveline side's — which is the hole the engagement logic above has: `caught`
// watches the road, and a full brake application arrests the wheels faster than any pedal follows,
// so the road-side answer arrives after the engine is already below its floor.
export [[nodiscard]] double antiStallPedal(const AutoClutch& assist, const double idleSpeed, const double engineSpeed)
{
    if (!assist.enabled)
    {
        return 0.0;
    }

    const auto begin = assist.antiStallBegin * idleSpeed;
    const auto open = assist.antiStallOpen * idleSpeed;

    return std::clamp((begin - engineSpeed) / std::max(begin - open, 1e-9), 0.0, 1.0);
}

// Where the pedal goes next, given where it is, what the driver is doing with it and what the
// automation would like.
//
// The driver's foot wins outright past the free play, and neither blend works. Taking the more
// released of the two lets the automation let out a clutch the driver is deliberately biting, so a
// human cannot launch the car at all; taking the more clamped lets it clamp one the driver is
// deliberately slipping, so a human cannot protect the engine. The automation is an assist and the
// driver's own foot is what switches it off, exactly as a brake pedal cancels cruise control — and
// the line between the two is the free play a real pedal already has rather than an invented
// threshold.
export [[nodiscard]] double advanceClutchPedal(const AutoClutch& assist, const double pedal, const double driverPedal,
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
