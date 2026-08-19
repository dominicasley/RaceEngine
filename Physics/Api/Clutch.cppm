module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>

#include <glm/glm.hpp>

export module raceengine.physics:Clutch;

import :Coupling;
import :Vehicle;

namespace raceengine
{

// The element between the engine's inertia and the gearbox input. It is an *element* rather than a
// clutch because a torque converter lands in the same slot next, and nothing downstream may know
// which is fitted: the driveline asks this for a torque and puts it on both sides, which is the same
// sentence either way.
//
// Two kinds behind one function and one output struct, which is `SuspensionKind`'s shape and
// deliberately not an abstract base's. What a driveline wants from this slot is a torque and whether
// it is held, and that does not vary by type; the pedal a friction clutch reads and the impeller
// speed a converter reads are already in the arguments. Value semantics throughout, so
// `DrivelineSetup` stays assignable.
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

// The other kind, stated and not implemented. The three numbers a converter is quoted by are here so
// that the slot's data is complete before anything reads them, and `stepDriveCoupling` refuses the
// kind outright rather than answering for it.
export struct TorqueConverter
{
    double stallTorqueRatio = 2.0;
    double capacityFactor = 150.0;
    double lockupSpeed = 0.0;
};

export struct DriveCoupling
{
    DriveCouplingKind kind = DriveCouplingKind::FrictionClutch;

    FrictionClutch clutch;
    TorqueConverter converter;
};

// One tick of whatever is fitted. `pedal` is the clutch pedal, 0 released and 1 fully depressed; a
// converter will ignore it, which is why the argument belongs to the slot rather than to the clutch.
//
// The converter case returns an error and does not fall through to the clutch's answer. A slot that
// quietly returned zero for it would be a car that will not move for no stated reason, and one that
// returned the clutch's answer would be a car that drives — which is worse, because it looks like
// the converter works.
export [[nodiscard]] std::expected<CouplingSolution, std::string>
stepDriveCoupling(const DriveCoupling& coupling, CouplingState& state, const CouplingSides& sides, const double pedal,
                  const double deltaTime)
{
    switch (coupling.kind)
    {
    case DriveCouplingKind::FrictionClutch:
    {
        auto loaded = sides;
        loaded.capacity = clutchCapacity(coupling.clutch, pedal);

        return stepCoupling(coupling.clutch.coupling, state, loaded, deltaTime);
    }
    case DriveCouplingKind::TorqueConverter:
        break;
    }

    return std::unexpected("a torque converter is fitted and there is no converter model yet: it is the next "
                           "element to land in this slot, and the friction clutch's answer is not its answer");
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
                                               const double automatic, const double deltaTime)
{
    const auto driver = std::clamp(driverPedal, 0.0, 1.0);

    // No rate limit on the driver: a real pedal answers an ankle directly, and dumping it is a thing
    // a driver is allowed to do and this model is supposed to have an opinion about.
    if (!assist.enabled || driver > assist.freePlay)
    {
        return driver;
    }

    // Because the state above has been tracking the driver's pedal all the while it was the
    // driver's, the automation picks up where the foot left it instead of stepping.
    const auto limit = std::max(assist.pedalRate * deltaTime, 0.0);

    return std::clamp(pedal + std::clamp(std::clamp(automatic, 0.0, 1.0) - pedal, -limit, limit), 0.0, 1.0);
}

} // namespace raceengine
