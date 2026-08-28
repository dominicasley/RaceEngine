module;

#include <cstdint>

export module raceengine.physics:Brakes;

import :Ambient;

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

    // How that coefficient changes with the disc's temperature, as a multiplier — which is fade.
    //
    // **Empty is the brake this model had until 2026-08-28: a coefficient that is the same at 20 °C
    // and at 700.** A curve that reads 1.00 everywhere is identical to the bit, which is what makes
    // switching this on provable rather than plausible.
    //
    // The provenance splits in two and the code must not blur them. **Below about 343 °C the plateau
    // is the pad's own specification**: SAE J866 marks a lining with two letters for two temperature
    // bands — 200-400 °F and 300-650 °F — and an OE pad's `FF` is 0.35-0.45 in *both*, which is a
    // statement that it does not fade across them. **Above that the tail is borrowed**, from
    // published SAE J2522 Fade I runs where friction falls from 0.32-0.34 to 0.24-0.28, and it is
    // nobody's measurement of this pad. docs/brake-thermal-brief.md.
    TemperatureCurve fade;
};

// The couple's friction at a stated disc temperature — the coefficient through the fade curve.
// Exactly `coefficient` for a couple that states no curve, and exactly `coefficient` anywhere on the
// curve's own plateau.
export [[nodiscard]] double frictionAtTemperature(const FrictionCouple& couple, const double celsius);

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

    // Its thickness and its mass, metres and kilograms, and whether it is internally vented.
    //
    // **None of the three appears in a brake torque and all three decide a temperature.** They are
    // here rather than on a separate thermal struct because they are properties of the same part:
    // a disc is a lever, a friction face and a lump of iron, and splitting the lump off would let a
    // car state a 340 mm disc that weighs what a 310 mm one does.
    //
    // The mass is the one worth sourcing rather than deriving. Manufacturers and their competitors
    // publish it per part, and what is left of it once the swept ring's geometry is accounted for is
    // the hat — which is the same trick the tyre's carcass mass uses.
    double discThickness = 0.030;
    double discMass = 10.7;
    bool discVented = true;

    // The hat's height, metres — how far the mounting flange sits behind the swept ring. Catalogued
    // per part alongside the diameter and the thickness, and it appears in no torque either: it is
    // the length of the neck the ring's heat has to travel down to reach the wheel, and it is
    // therefore most of what decides whether stage 3's path is worth anything. docs/brake-thermal-brief.md.
    double hatHeight = 0.050;

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

// --- the disc as a lump of iron ---

// What one corner's disc is worth thermally, derived from the part above and grey iron's own
// properties. Everything here has a unit and none of it is fitted.
export struct BrakeThermal
{
    // J/K. The whole disc, hat included: the hat is bolted to the ring and conducts, and separating
    // them needs a joint conductance nobody publishes.
    double heatCapacity = 0.0;

    // m². What convects and what radiates are **not the same area**, and a vented disc is why: its
    // internal passages roughly double the area air passes over and radiate almost entirely to
    // themselves, so they cool by convection and not by radiation.
    double convectionArea = 0.0;
    double radiationArea = 0.0;

    // The diameter the Reynolds number is taken on, metres. The disc's own.
    double diameter = 0.340;

    // Emissivity of oxidised grey iron. Published band 0.7-0.9; a brake disc is oxidised within a
    // day of being fitted, so the bright-iron figure is the wrong end of it.
    double emissivity = 0.8;

    // What share of the friction power at the rubbing face heats the **disc** rather than the pad.
    //
    // **Derived rather than fitted, by the same partition the tyre uses.** Heat released at the plane
    // between two bodies divides in proportion to their thermal effusivities `sqrt(k·rho·c)`: grey
    // iron is 12868 and an organic pad about 1732, so the disc takes 0.881. Brake literature quotes
    // 90-95% into the disc, so this lands just under a band it was not fitted to.
    double heatToDisc = 0.881;

    // What share of `radiationArea` sees the **wheel** rather than the sky.
    //
    // **A partition and not an addition, because the alternative creates energy.** The disc's
    // outboard face is a few centimetres from the wheel's inner dish and radiates almost entirely to
    // it; the inboard face sees the upright, the shield and the road. Spending the same area twice —
    // once against ambient and once against the wheel — would cool the disc through a surface it
    // does not have.
    //
    // **Zero is a disc with no wheel stated**, which is every car in this project bar the Golf, and
    // zero reproduces the pre-stage-3 arithmetic exactly. `wheelThermalOf` sets it; the bound is
    // 0.4-0.6 and 0.5 is the geometry of two coaxial faces of similar radius a few centimetres
    // apart. docs/brake-thermal-brief.md.
    double wheelRadiationShare = 0.0;

    // The couple, carried so that a corner holding one of these holds the whole thermal brake: the
    // fade curve is a property of the pad-and-rotor pair and the temperature that reads it is a
    // property of the disc. `brakeThermalOf` copies it off the hardware, so it is stated once.
    FrictionCouple couple{};
};

// Grey cast iron, textbook, and the bands recorded beside the values taken:
//   density        7200 kg/m³
//   specific heat   460 J/(kg·K)   band 450-540
//   conductivity     50 W/(m·K)    band 45-55
export inline constexpr double castIronDensity = 7200.0;
export inline constexpr double castIronSpecificHeat = 460.0;
export inline constexpr double castIronConductivity = 50.0;

// What every brake disc in this project starts at, degrees Celsius.
//
// **Cold, and cold is the inert seed here** — which is the opposite of the tyre's and is the more
// interesting half of the symmetry. A pad's rated friction is flat from 93 °C to 343 and this model's
// fade curve is flat below it, so a disc seeded cold multiplies brake torque by exactly one; the
// tyre's grip curve is flat only in its *hot* plateau, so its state had to be seeded warm to get the
// same proof. Both switches are inert at their own curve's flat part and the two are at opposite
// ends. This one is also the physical seed: a car in a garage has cold brakes.
export inline constexpr double brakeDefaultTemperature = 20.0;

// The swept ring's one-face area, m² — the annulus the pad rubs, from the disc's diameter and the
// pad's radial height. The same two numbers `effectiveRadius` is built from.
export [[nodiscard]] double sweptFaceArea(const BrakeHardware& brake);

// The thermal half of a brake, derived from its parts. Nothing here is stated twice: the mass, the
// diameter, the pad's geometry and the couple all come off the hardware.
export [[nodiscard]] BrakeThermal brakeThermalOf(const BrakeHardware& brake);

// What one disc is doing thermally over one tick.
export struct BrakeThermalInput
{
    // Watts at the rubbing face — the brake torque actually applied times the speed it was applied
    // at. **Applied and not commanded**: a locked wheel makes no heat at the disc, it makes it at the
    // road, and the tyre model already has that.
    double frictionPower = 0.0;

    // How fast the car is going through the air, m/s. The disc's convection is forced and this is
    // what forces it.
    double airSpeed = 0.0;

    // The wheel this disc is bolted inside, and how strongly the two are coupled, W/K — conduction
    // through the hat and the bolted joint *plus* the radiation across the gap, as one number.
    //
    // **It is one number because the two nodes must not be able to disagree about it.** The caller
    // computes it once with `discToWheelCoupling` and hands the same value to this step and to
    // `stepWheelThermal`, so whatever leaves the disc arrives at the wheel.
    //
    // **Zero is the brake stages 1 and 2 shipped**, which had no path out through the wheel at all.
    double wheelTemperature = brakeDefaultTemperature;
    double wheelConductance = 0.0;

    AmbientConditions ambient;
};

// One tick of one disc's heat balance, advancing `celsius`.
//
// Integrated exactly against a linearised loss, which is `relaxTyre`'s treatment for its reason: the
// closed form is unconditionally stable and cannot stretch a time constant with the timestep.
// Radiation is linearised about the current temperature within the tick, which is where a fourth
// power is indistinguishable from a straight line.
export void stepBrakeThermal(const BrakeThermal& thermal, double& celsius, const BrakeThermalInput& input,
                             const double deltaTime);

// The disc's forced-convection coefficient, W/(m²·K) — **Limpert's rotor correlation**,
// `h = 0.04 · (k_air / D) · Re^0.8` with `Re = V·D/nu`.
//
// The same functional form the tyre's Hilpert relation has, and it reads air's conductivity and
// viscosity out of `:Ambient` for that reason. Anchors: an independent analysis of a passenger disc
// lands at 58-60 W/(m²·K) at 60 km/h against CFD, and this correlation gives about 87 there — it is
// known to sit high because it is a whole-assembly figure that carries the vanes' own pumping.
export [[nodiscard]] double discConvection(const BrakeThermal& thermal, const double airSpeed, const double celsius,
                                           const double airCelsius);

// --- the wheel the disc is bolted inside -------------------------------------------------------

// A356 cast aluminium, textbook, and the bands recorded beside the values taken:
//   density        2600 kg/m³
//   specific heat   900 J/(kg·K)   band 900-963 (the upper figure is at 100 °C)
//   conductivity    150 W/(m·K)    band 149-155
export inline constexpr double castAluminiumDensity = 2600.0;
export inline constexpr double castAluminiumSpecificHeat = 900.0;
export inline constexpr double castAluminiumConductivity = 150.0;

// The wheel, as parts, on the same principle the disc is stated with.
//
// **This is the third node on the path the tyre work asked for**: a disc runs 200-550 °C in ordinary
// hard driving and a tread core runs 50, and until stage 3 there was nothing between them. What sits
// between them is a large lump of aluminium in the airstream, and the whole question is how much of
// what crosses into it reaches the tyre rather than leaving to the air.
//
// **A zero mass is "no wheel stated"**, which is every car in this project except the Golf and is
// what makes the whole of stage 3 inert by default: no capacity, no conductances, no radiation
// share, and a disc whose arithmetic is bit-for-bit what stages 1 and 2 shipped.
export struct WheelHardware
{
    // Kilograms. **Sourced, and a class figure rather than the OE part's**, on the same footing as
    // the tyre's published mass: a direct replacement for this car's own 18 × 7.5 5×112 57.1 mm
    // wheel is quoted at 27 lb, which is 12.25 kg. A reproduction usually runs a little heavier than
    // the casting it replaces, so the bias is known and is toward a slower wheel.
    double mass = 0.0;

    // The wheel's outside diameter, metres — an 18 inch rim. The length scale the Reynolds number
    // is taken on, exactly as the disc's own diameter is.
    double diameter = 0.4572;

    // Of the painted inner face, which is the one that faces the disc. Bare aluminium is under 0.03
    // and would make this path nearly nothing; paint runs near unity. Every wheel's brake-side face
    // is painted or primed, and machined faces are on the outside where no disc can see them.
    double emissivity = 0.85;

    // --- what sets the conduction, and the surprise is that the joint does not ---
    //
    // The hat is a thin iron neck between the swept ring and the mounting flange, and it is in
    // series with the bolted joint. **The neck is the bottleneck by a factor of thirty** — which is
    // why the quantity this brief called blocking turns out not to decide anything.
    //
    // Wall thickness in metres, bounded at 5-9 mm for a 340 mm hat; and the bolt circle's radius,
    // which is published as a PCD (5 × 112 gives 0.056).
    double hatWallThickness = 0.007;
    double boltCircleRadius = 0.056;

    // W/K across the bolted joint itself. **The one quantity in either thermal model that nobody
    // publishes**, and it is bounded rather than fitted.
    //
    // Hasselström, *Thermal Contact Conductance in Bolted Joints* (Chalmers, 2012), Table C.1
    // measures aluminium 6082 at Ra 0.8 µm in vacuum: 1509 W/(m²·K) at 2.0 MPa rising to 112994 at
    // 24.6, against Fletcher & Gyorog's correlation over the same range at 1459 to 7033. Five M14
    // bolts at 120 N·m put about 43 kN each through a contact annulus of roughly 7.5 to 22 mm of
    // radius — 6.7e-3 m² at 32 MPa — which is **60 to 760 W/K** depending on which of two published
    // estimates is believed.
    //
    // **The low end is taken and it costs 3%.** In series with a 2.09 W/K hat neck the whole path is
    // 2.015 W/K at 60 and 2.079 at 760: a thirteen-fold uncertainty in the unpublished number moves
    // the answer by one part in thirty. Vacuum figures are also a floor, because air conducts across
    // the gaps a vacuum leaves empty.
    double jointConductance = 60.0;

    // W/K from the wheel into the tyre's **carcass**, bead seats and inner air together.
    //
    // Two paths in parallel, both derived. The beads: two seats of about 20 mm on an 18 inch
    // circumference is 0.057 m², and the resistance is the bead rubber's own over half its depth —
    // `0.209 / 0.010` = 21 W/(m²·K), so 1.2 W/K. The cavity: the rim well's 0.27 m² and the liner's
    // 0.6 m² in series through air stirred by rotation at a bounded 10-30 W/(m²·K), which is 2.8.
    // Together **4.0 W/K**, and the two are the same order, so neither can be dropped.
    //
    // It reaches the carcass and not the tread, which is why the answer is small: the carcass loses
    // to the air over both sidewalls before the core sees any of it.
    double toTyre = 4.0;

    // What share of the disc's radiating area sees this wheel. See `BrakeThermal::wheelRadiationShare`
    // — the partition lives there because it is the disc's area being divided.
    double discRadiationShare = 0.5;
};

// What one corner's wheel is worth thermally, derived from the parts above and from the disc it is
// bolted to. Everything has a unit and none of it is fitted.
export struct WheelThermal
{
    // J/K.
    double heatCapacity = 0.0;

    // m². Both faces of a disc of the wheel's own diameter. A cast wheel has more surface than that
    // and less of it is exposed, standing inside an arch behind a tyre, and the two roughly cancel —
    // stated as a bound rather than claimed as a measurement.
    double convectionArea = 0.0;

    // m². The same, less the patch the disc is already radiating onto — otherwise that patch would
    // be spent twice, once against the disc and once against the sky, which is the mistake the
    // disc's own partition exists to avoid.
    double radiationArea = 0.0;

    // The diameter the Reynolds number is taken on, metres.
    double diameter = 0.4572;

    double emissivity = 0.85;

    // W/K. The hat's neck in series with the bolted joint — conduction only. The radiation across
    // the gap is temperature-dependent and is `discToWheelRadiation`'s.
    double toDisc = 0.0;

    // W/K into the tyre's carcass.
    double toTyre = 0.0;
};

// The hat's own conduction from the swept ring to the mounting flange, W/K — geometry and grey
// iron's conductivity, and nothing else.
//
// A cylindrical wall of `hatWallThickness` from the swept ring's inner radius, running the hat's
// height, and then a radial annulus inward to the bolt circle. The hat's height is taken as the
// disc's own catalogued figure. **This is the number the bolted joint is in series with, and it is
// thirty times smaller.**
export [[nodiscard]] double hatConductance(const BrakeHardware& brake, const WheelHardware& wheel);

// The thermal half of a wheel, from its parts and the disc it is bolted to. A wheel of zero mass
// returns a node with zero capacity and zero conductances, which every step below treats as absent.
export [[nodiscard]] WheelThermal wheelThermalOf(const WheelHardware& wheel, const BrakeHardware& brake);

// The disc, told which wheel it is bolted inside — `brakeThermalOf(brake)` and then the radiation
// partition. **An overload rather than a defaulted argument**, so that every existing caller keeps
// the one-argument form and a disc with no wheel is not merely equal to the old one but is the same
// expression.
export [[nodiscard]] BrakeThermal brakeThermalOf(const BrakeHardware& brake, const WheelHardware& wheel);

// The radiative conductance between a disc and the wheel it is bolted inside, W/K.
//
// Linearised about the two current temperatures exactly as the disc's radiation to the sky is —
// `h = eps·sigma·(T1² + T2²)(T1 + T2)` is the exact secant of the fourth power between them, so what
// is approximated is only how far either moves inside one tick.
//
// **The effective emissivity is the two-surface one**, `1/(1/e1 + 1/e2 - 1)`, because these are two
// close, nearly parallel faces re-radiating at each other rather than one body facing a sky. Oxidised
// iron at 0.8 and painted aluminium at 0.85 give 0.701.
export [[nodiscard]] double discToWheelRadiation(const BrakeThermal& disc, const WheelThermal& wheel,
                                                 const double discCelsius, const double wheelCelsius);

// The whole coupling, W/K: `wheel.toDisc` plus the radiation above. **One number, computed once by
// the caller and handed to both steps**, so that what leaves the disc is what arrives at the wheel.
export [[nodiscard]] double discToWheelCoupling(const BrakeThermal& disc, const WheelThermal& wheel,
                                                const double discCelsius, const double wheelCelsius);

// What one wheel is doing thermally over one tick.
export struct WheelThermalInput
{
    // The disc, and the coupling `discToWheelCoupling` returned — the same value the disc's own step
    // was given.
    double discTemperature = brakeDefaultTemperature;
    double discConductance = 0.0;

    // The tyre's carcass, and the conductance into it. **Zero whenever the tyre carries no
    // temperature**, because a tyre that is not being simulated has nothing to exchange with and
    // pretending otherwise would let the wheel warm against a constant.
    double tyreTemperature = 0.0;
    double tyreConductance = 0.0;

    // How fast the wheel is going through the air, m/s.
    double airSpeed = 0.0;

    AmbientConditions ambient;
};

// One tick of one wheel's heat balance, advancing `celsius`. Integrated exactly against a linearised
// loss, which is `stepBrakeThermal`'s treatment and `relaxTyre`'s before it.
export void stepWheelThermal(const WheelThermal& thermal, double& celsius, const WheelThermalInput& input,
                             const double deltaTime);

// The wheel's forced-convection coefficient, W/(m²·K) — **Limpert's rotor correlation again**, on the
// wheel's own diameter. It is the same object the correlation was written for: a disc of that
// diameter turning in an airstream. It is known to sit high, which here under-states the effect this
// stage is measuring rather than flattering it.
export [[nodiscard]] double wheelConvection(const WheelThermal& thermal, const double airSpeed, const double celsius,
                                            const double airCelsius);

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
