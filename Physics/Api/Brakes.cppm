module;

#include <cstdint>

export module raceengine.physics:Brakes;

namespace raceengine
{

// A brake, from the parts that make it rather than from one scalar.
//
// The scalar it replaces was `CornerSetup::brakeTorque`, and its whole history is why this partition
// exists: `brakes.ini` stated a `MAX_TORQUE` that was identical across four different cars in the
// same pack, could not lock this car's front wheels at any pedal position, and was replaced on
// 2026-08-23 by **another number somebody chose** — 5600, marked at the time as a placeholder above a
// derived 4624 bound. A number nobody chose is the point of this file.
//
// **The split is offline/online and it is Dominic's**: *"pressure x area of brake pad x friction of
// the pad and its distance from the centre of the hub — these could be an offline function that only
// needs to be calculated when the rotor/caliper change"*. Everything geometric and material is
// constant while the hardware is, so it collapses to one number — `torquePerPressure`, N.m per pascal
// — computed once at load. The tick multiplies by pressure and does nothing else, which is cheaper
// than what was there before.

// Friction is a property of the **couple** and not of the pad, which is why this is a type and not a
// double on the pad. The same compound reads differently on grey iron and on carbon-ceramic, and pads
// are specified against a rotor type for that reason; carrying two numbers and combining them would
// produce confident figures for pairings that do not behave that way.
export struct FrictionCouple
{
    double coefficient = 0.40;
};

// An OE-class organic/low-metallic pad on a grey cast iron rotor, which is what every disc on this
// car is.
//
// **Sourced to the pad's own edge code rather than to a plausible value.** SAE J866 marks a lining
// with two letters for its cold and hot friction, in bands of 0.10: E is 0.25-0.35, F is 0.35-0.45,
// G is 0.45-0.55. OE passenger-car pads are marked **FF** almost without exception — the same
// coefficient cold and hot, which is what "no surprises for the driver" means in a friction spec —
// and the standard's own test runs against a **cast iron** drum, so the band is already a statement
// about this couple and not about the pad alone. 0.40 is the midpoint of F.
//
// The uncertainty this carries is +/-0.05 on 0.40, which is +/-12.5% straight into every torque
// below. That is the largest single uncertainty in the derivation and it is worth knowing where it
// is: it is not the geometry, all of which is measured to the millimetre.
export [[nodiscard]] FrictionCouple lowMetallicOnCastIron();

// One corner's brake, as parts.
//
// **A sliding caliper clamps both pads with one piston**, which is why `frictionFaces` is 2 while
// `pistons` is 1: the piston pushes the inboard pad onto the disc and the reaction drags the caliper
// body inboard, pressing the outboard pad on with the same force. Doubling the piston count for a
// sliding caliper — which is the obvious mistake — doubles the whole car's brake torque.
export struct BrakeHardware
{
    // Piston bore, metres, and the count of pistons on **one side** of the disc.
    double pistonBore = 0.060;
    std::uint32_t pistons = 1;

    // The disc's outside diameter, metres.
    double discDiameter = 0.340;

    // The pad's radial height, metres — its dimension from the inner edge of the swept annulus to
    // the outer. Not the pad's other two dimensions: the circumferential length and the thickness
    // set the wear rate and the thermal mass, and neither appears in a torque.
    double padRadialHeight = 0.070;

    // How far inside the disc's outer edge the pad's outer edge sits, metres. Small, and it is the
    // one geometric figure here nobody has measured: 3-5 mm is universal practice, because a pad
    // swept right to the edge leaves a lip that fouls the next set. 5 mm moves the effective radius
    // by 5 mm, which is 3.8% of it on the front — the largest of the geometric uncertainties and
    // still a third of the friction coefficient's.
    double padOuterClearance = 0.005;

    // A disc has two sides. Stated rather than assumed because a drum does not, and because this is
    // where a drum brake or an inboard disc would say so.
    std::uint32_t frictionFaces = 2;

    FrictionCouple couple{};
};

// Total piston area on one side of the disc, m^2.
export [[nodiscard]] double pistonArea(const BrakeHardware& brake);

// The mean radius of the pad's swept annulus, metres — the lever the friction force acts on.
//
// `(outer + inner) / 2`, which is the standard approximation and is what a pad of roughly constant
// circumferential width gives. The area-weighted radius of a true annular *sector*,
// `(2/3)(Ro^3 - Ri^3)/(Ro^2 - Ri^2)`, is 3 mm larger on this car's front pad; a real pad is closer to
// a rectangle than to a sector, so the mean is the better of the two and the difference is well
// inside the friction coefficient's band either way.
export [[nodiscard]] double effectiveRadius(const BrakeHardware& brake);

// The whole offline half, N.m of brake torque per pascal of line pressure. Everything above,
// multiplied: piston area, the couple's friction, the effective radius and the number of faces.
export [[nodiscard]] double torquePerPressure(const BrakeHardware& brake);

// What fraction of the car's brake torque the front axle makes, given the two corners' hardware.
//
// **This is what stops `FRONT_SHARE` being a number.** A brake split is not a setting on a car
// without a proportioning valve: it is the ratio of two `torquePerPressure` figures, because both
// axles see the same line pressure. Anything that changes it — a bigger front piston, a smaller rear
// disc, a different pad on one axle — changes it through the parts.
export [[nodiscard]] double frontBrakeShare(const BrakeHardware& front, const BrakeHardware& rear);

// The other half: what the driver's foot does to the line pressure. Per car, not per corner — one
// master cylinder feeds both circuits, which is exactly why the split above is hardware and not a
// choice.
export struct BrakeHydraulics
{
    // Master cylinder bore, metres.
    double masterCylinderBore = 0.02381;

    // The pedal's mechanical advantage: pushrod force per unit of driver force.
    double pedalRatio = 3.5;

    // The vacuum servo's gain, as output force per unit of input force — so 1.0 is a car with no
    // servo, and the assist is `boostRatio - 1` times the input.
    double boostRatio = 4.0;

    // And what it runs out at, stated as the parts that set it: the diaphragm's effective diameter
    // in metres and the depression across it in pascals. Past their product the servo has nothing
    // left to give and the gain drops to 1 — the driver's own foot and nothing else. **The runout is
    // not a detail**: it is why a real brake pedal goes hard near the floor, and it is what keeps the
    // peak below from being a straight multiplication of two guessed ratios.
    // **No servo fitted, which is the default and not this car.** A diaphragm of zero makes the
    // assist limit zero, so `boostRatio` above is inert and the pressure is linear in the pedal — the
    // same map every car in this project had before the hydraulics existed, so a car that states no
    // brake system brakes identically to the bit. The Golf states its own; see `golfMk7Hydraulics`.
    double boosterDiaphragm = 0.0;
    double boosterVacuum = 75000.0;

    // What a fully applied pedal means, newtons at the driver's foot.
    //
    // **The one number here that is a definition rather than a part**, and it is worth being loud
    // about because it is where "how much pedal does locking take" now lives. `VehicleInput::brake`
    // is a demand from 0 to 1 with no units; turning it into a force needs a statement of what 1
    // means, and 500 N is UN ECE R13-H's control-force ceiling for the M1 service-brake tests — the
    // hardest push any regulation expects of a driver. Lower it and the car locks later on the pedal;
    // it does not change a single thing about the brakes.
    double maxPedalForce = 500.0;
};

// The **other half of the brake bias**, and the half a caliper-and-disc model does not have.
//
// `frontBrakeShare` above is the whole of what the hardware makes of one line pressure. A real car
// does not stop there: it biases the *pressure* as well, because no fixed split can be right across a
// stop. Load transfer moves the ideal split with deceleration — on this car from 0.647 at 0.3 g to
// about 0.81 where it locks — so a split chosen for the middle of that range locks the rear axle
// first at the top of it, which is the one failure order no road car is set up with.
//
// Every car answers this, and there are three ways: a **fixed proportioning valve** in the rear line,
// which is what this is; a **load-sensing** valve, which is the same thing with the knee moved by
// suspension travel; or **EBD**, which is the anti-lock unit doing it in software and is what a Mk7
// actually has. The first is the one that can be stated as hardware, and it is the honest stand-in
// for a model whose anti-lock unit has no EBD in it.
//
// The characteristic is two straight lines: rear pressure follows the master cylinder up to
// `kneePressure` and then rises at `slope`. That is a two-point approximation to the ideal
// distribution curve, which is exactly what the real component is.
//
// **`slope = 1` is a car with no valve fitted**, whatever the knee says, and it is the default — so
// nothing that does not state one changes by a bit.
export struct ProportioningValve
{
    double kneePressure = 0.0;
    double slope = 1.0;
};

// What the rear circuit sees, given what the master cylinder is at. Both in pascals.
export [[nodiscard]] double proportionedPressure(const ProportioningValve& valve, const double inlet);

// Master cylinder area, m^2.
export [[nodiscard]] double masterCylinderArea(const BrakeHydraulics& hydraulics);

// The most force the servo can add, newtons — depression times diaphragm area. The pushrod's own
// area is subtracted by a real calculation and is under 1% of the diaphragm's, so it is not here.
export [[nodiscard]] double boosterAssistLimit(const BrakeHydraulics& hydraulics);

// Line pressure at a given pedal demand, pascals. Two straight lines with a knee at the servo's
// runout, which is the shape a boosted system's own characteristic has.
export [[nodiscard]] double brakeLinePressure(const BrakeHydraulics& hydraulics, const double pedal);

// The pedal demand at which the servo runs out, 0 to 1. Above 1 for a servo big enough never to run
// out inside the pedal's range, which is a legitimate design and is why this is not clamped.
export [[nodiscard]] double boosterRunoutPedal(const BrakeHydraulics& hydraulics);

// One corner's brake torque at a given pedal, N.m. The whole model in one line, and the only thing a
// tick would ever need.
export [[nodiscard]] double brakeTorqueAtPedal(const BrakeHardware& brake, const BrakeHydraulics& hydraulics,
                                               const double pedal);

// What that corner makes at a fully applied pedal — which is what `CornerSetup::brakeTorque` is, and
// therefore the whole of what this partition currently delivers to the vehicle model.
//
// **Stage 1 of `docs/brake-model-brief.md` deliberately stops here.** The tick still multiplies a
// peak by a pedal, so the pressure curve above is exercised at exactly one point; moving the tick and
// the anti-lock modulator into real pressure units is Stage 2, and it is what closes the cycling-rate
// red in `docs/known-red.md`. Deriving the peak first is the smaller change and it is the one that
// deletes a chosen number.
export [[nodiscard]] double peakBrakeTorque(const BrakeHardware& brake, const BrakeHydraulics& hydraulics);

} // namespace raceengine
