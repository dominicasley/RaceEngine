module;

// Only what the *declarations* below need. <algorithm> and <cmath> moved to TyreImpl.cpp with the
// bodies that use them — see the header of that file for why the bodies are no longer here.
#include <cstdint>
#include <type_traits>

export module raceengine.physics:Tyre;

import :Ambient;

namespace raceengine
{
// The tire, as a Magic Formula with the two things that decide whether a car feels like a car:
// load sensitivity, and relaxation length. Everything else here is arithmetic around them.
//
// Pure and stateless except for the two carcass deflections it is asked to carry, so it can be
// swept, plotted and argued about without a chassis anywhere near it.

// Which of the tyre's two directions a question is being asked about. It exists so that the peak and
// the load-sensitivity exponent — of which there are now two apiece — can only ever be read as a
// matched pair.
export enum class TyreAxis : std::uint32_t { Lateral, Longitudinal };

// The tread's heat balance, stated so that **every constant carries a unit** and can be sourced,
// derived from the tyre's own geometry, or bounded. That is the whole design rule here, and it is
// the reason AC's own thermal gains are not used: `FRICTION_K = 0.0619` and its four companions
// carry no unit, no timestep, no area and no mass, so they are numbers without a convention and a
// number without a convention is not a measurement. docs/tyre-state-brief.md, section 2.
//
// **Three nodes and one direction of travel.** Heat arrives at the *surface* from sliding at the
// patch and in the *tread core* and *carcass* from the carcass's own hysteresis; it leaves by
// convection to the air and by conduction into the road. The grip curve reads the **core**, not the
// surface, and that is sourced rather than assumed — Farroni, Russo, Sakhnevych and Timpone,
// *TRT EVO*, Proc IMechE Part D 233(1) 2019: "the tire frictional behavior can be more realistically
// associated with the tread core layer temperature", because friction is set by the bulk
// viscoelastic state of the tread block and a thin skin cannot change it fast enough to matter.
export struct TyreThermal
{
    // --- geometry, which is this tyre's own size and nothing else ---
    //
    // The tread band's outer radius and the rim's, metres. For a 225/40 R18 those are 0.2286 (an 18
    // inch rim) plus 0.090 of section height (40% of 225 mm) and the rim radius itself, which is
    // where `golfTyreRadius = 0.3186` comes from independently.
    double outerRadius = 0.33;
    double rimRadius = 0.2286;
    // The tread band's width, metres.
    double treadWidth = 0.20;

    // New-tyre groove depth and the undertread below it, metres. The first is a published figure per
    // tyre; the second is the rubber between the groove floor and the belt, which no manufacturer
    // publishes and which is 1.5 to 3 mm on a passenger radial.
    double grooveDepth = 0.0075;
    double underTread = 0.002;

    // What fraction of the grooved band is groove rather than rubber. A performance summer tread
    // runs about 25-30% void; it is bounded rather than sourced per pattern.
    double voidFraction = 0.28;

    // The whole tyre's mass, kilograms. **This is what makes the carcass node's heat capacity a
    // derived quantity rather than a guess**: the tread's mass follows from the geometry above, and
    // what is left of a published tyre mass is everything else — belt, plies, liner, sidewalls and
    // beads. Manufacturers publish it per size.
    double tyreMass = 10.0;

    // How thick the surface node is, metres. **A discretisation and not a physical constant**, and
    // it is chosen from the material: the thermal diffusion length in tread rubber over the couple of
    // seconds a corner lasts is `sqrt(alpha·t)` = 0.45 mm at this diffusivity, so half a millimetre
    // is the depth that genuinely responds inside one corner. Making it thicker slows the surface
    // channel and does not move the core, which is what grip reads.
    double surfaceThickness = 0.0005;

    // Half the belt-and-ply package's depth, metres — how far below the tread the carcass node's
    // centroid sits. A construction estimate in the 4 to 8 mm band for a passenger radial's belt
    // package, and it sets one conductance and nothing else.
    double carcassHalfDepth = 0.003;

    // --- material, sourced ---
    //
    // Tread rubber, from Clark, *Heat Generation in Aircraft Tires*, NASA 1983, whose three figures
    // are mutually consistent: they give a diffusivity of 1.0e-7 m²/s, which is the textbook value
    // for rubber and is the check that the trio belongs together.
    //
    // **Conductivity is the loosely constrained one and it matters least.** An independent NR/SBR
    // tread compound measures 0.56 W/(m·K) against Clark's 0.209 — a factor of 2.7 — while the two
    // densities agree to 5%. Volumetric heat capacity `density · specificHeat` is what turns watts
    // into a rate of temperature change and therefore sets the warm-up; conductivity only sets how
    // fast the nodes equalise with each other. **The stated range is 0.21 to 0.56** and Clark's value
    // is taken because his three numbers are self-consistent.
    double conductivity = 0.209;
    double density = 1000.0;
    double specificHeat = 2092.0;

    // The road's thermal effusivity, `sqrt(k·rho·c)`, J/(m²·K·s^0.5). Asphalt concrete's textbook
    // trio — 1.2 W/(m·K), 2300 kg/m³, 900 J/(kg·K) — gives 1576, and it enters in exactly one place:
    // two bodies brought into contact settle at an effusivity-weighted mean rather than at either
    // one's temperature, so this is what says how much of the difference the tread actually feels.
    double roadEffusivity = 1576.0;

    // The thermal contact conductance of the tread-road interface itself, W/(m²·K), **in series**
    // with the semi-infinite solution above: `1/h = 1/h_perfect + 1/h_contact`. **Zero is perfect
    // contact**, which is what the model did before this existed and is bit-identical to it.
    //
    // It is here because the semi-infinite solution assumes the two surfaces touch everywhere, and
    // rubber on rough asphalt does not — only the asperity tips carry heat. The tyre brief ranked
    // that assumption as the last candidate that could explain a tread running twenty degrees below
    // its window, and expected it to halve the road path.
    //
    // **It is measured rather than derived, on this exact interface, and it says the effect is a
    // fifth rather than a half.** C. David Miller, *Thermal Conductance of and Heat Generation in
    // Tire-Pavement Interface and Effect on Aircraft Braking*, NASA TN D-8161, 1976: a
    // finite-difference analysis of temperature records from a free-rolling automotive tyre at
    // 22.35 m/s and from thin-film thermometers cemented to the pavement it rolled over. He measures
    // **3 × 10⁴ W/(m²·K) for rubber against the sensor's polyimide** — bounded below by 1.2 × 10⁴
    // and possibly at or above 5.7 × 10⁴ — and converts it through the two materials'
    // conductivities to **2.52 × 10⁴ for rubber against asphalt**, which is this figure. **He states
    // it as a lower limit**, so the true resistance is at most this and probably less.
    //
    // His own explanation is the answer to the question the term was added to ask: *"This is a very
    // high conductance for solids in contact. A conductance of this order of magnitude probably
    // exists only because the rubber under pressure deforms to make molecular contact over a large
    // fraction of the supporting surface in spite of asperities on that surface."*
    //
    // **The series form is an approximation and its size is stated rather than hidden.** The exact
    // transient for two semi-infinite bodies joined by an interface conductance is
    // `h(t) = h_c·exp(b²t)·erfc(b·sqrt(t))` with `b = h_c·(e₁+e₂)/(e₁·e₂)`, averaging over the
    // residence time to `h_c·[e^τ·erfc(sqrt(τ)) − 1 + 2·sqrt(τ/pi)]/τ`. At this conductance and a
    // 100 km/h residence that is 5242 W/(m²·K) against the series form's 5051 — **the series is
    // 3.4% pessimistic**, which is 0.7% of the road path. It is taken because it keeps the unstated
    // case exactly the old expression, and because it errs *cool*, which is the direction this model
    // resolves uncertainty in. `TyreThermalTests` pins the gap so it cannot grow unnoticed.
    double roadContactConductance = 0.0;

    // Still-air convection from a cylinder, W/(m²·K). A floor under the forced-convection
    // correlation rather than a term of its own: the Hilpert relation is a *forced* correlation and
    // goes to zero with speed, and a parked car still cools. The textbook band for free convection in
    // air is 5 to 10 and this is the low end, which is the conservative choice for a cooling term.
    double naturalConvection = 5.0;

    // What share of the power dissipated at the sliding contact patch heats the **rubber** rather
    // than the road.
    //
    // **The brief that planned this work called this the one fitted number in the whole model, and
    // it turns out to be derivable from two numbers already here.** Heat released at the plane
    // between two semi-infinite bodies divides between them in proportion to their thermal
    // effusivities, so the tread's share is `e_tread / (e_tread + e_road)` — and both effusivities
    // are already stated above, the first as Clark's `sqrt(k·rho·c)` = 661 and the second as
    // asphalt's 1576. That gives **0.2956**, which is what this is.
    //
    // It does not double-count the road conduction in `stepTyreThermal`. The heat equation is
    // linear, so the interface problem superposes: a *source* at the plane with both bodies cold,
    // which divides by effusivity, plus *no source* with the two bodies at different temperatures,
    // which is the contact conduction. They are two different terms of one solution and both belong.
    //
    // **It is a lower bound and the reason is worth keeping.** The partition above assumes the heat
    // appears exactly at the plane, and a real tyre's friction is substantially *hysteretic* — the
    // rubber losing energy as it is deformed over the asperities, a fraction of a millimetre down
    // rather than at the surface. Every joule generated that way is already in the rubber. Nobody
    // publishes the split for rubber on asphalt, so the conservative end is taken: it makes the tyre
    // run cooler, and cooler is the direction that makes this car's open braking defect worse rather
    // than better, which is the right way for an uncertainty to be resolved.
    double frictionToTread = 0.2956;

    // --- what the temperature is worth ---
    //
    // Grip against the **core** temperature, dimensionless, multiplying `TyreModel::gripScale`.
    TemperatureCurve grip;

    // Where this compound wants to be, degrees Celsius. Not read by the model — the curve is what
    // acts — and stated so that a fixture, a probe or a seat knob can ask a car for its own ideal
    // rather than spelling a number that belongs to a compound.
    double idealTemperature = 85.0;
};

// Standard atmospheric pressure, pascals. The gas law needs absolute pressure and every tyre
// pressure anybody quotes is a gauge one, so the conversion happens once and here.
export inline constexpr double atmosphericPressure = 101325.0;

// One pascal in psi, for the places where a human number has to be read or printed. Tyre pressures
// are quoted in psi or bar everywhere in the sources, and stored in pascals like every other
// pressure in this model.
export inline constexpr double psiPerPascal = 1.0 / 6894.757;

// The air inside the tyre, and what its pressure is worth to the car.
//
// **The gas law is the whole model and it has no constant to choose.** A tyre is a fixed volume, so
// absolute pressure follows absolute temperature: set 28 psi in a garage at 20 °C and the same tyre
// reads 34 psi when its air reaches 61 °C. That is not a coincidence with `PRESSURE_IDEAL 34` — it
// is what that number *means*, and it is why pressure had to come after temperature rather than
// before it.
//
// **Two couplings act, and neither imports a fitted number.** The carcass's vertical rate is linear
// in gauge pressure, which is the constant-belt result — Rhyne, *Development of a Vertical Stiffness
// Relationship for Belted Radial Tires*, Tire Science and Technology 33(3):136, 2005, which derives
// stiffness from the contained air in an inextensible ring — and AC's own `PRESSURE_SPRING_GAIN`
// agrees with a line through the origin to 1.1%. Rolling resistance falls as a power of pressure,
// because a softer tyre bends its sidewalls further and hysteresis follows the deformation.
//
// **Grip against pressure has a sourced form and no sourced number, and those are two different
// problems.** The Magic Formula's own pressure extension multiplies a peak friction coefficient by
// `1 + p3·dpi + p4·dpi²` against a dimensionless `dpi = (P − P_ideal)/P_ideal` — Besselink, Schmeitz
// & Pacejka, *An improved Magic Formula/Swift tyre model that can handle inflation pressure changes*,
// Vehicle System Dynamics 48(sup1):337, 2010. So the shape is published and carries a convention,
// unlike AC's `PRESSURE_D_GAIN`, which has no unit and no reference exactly as its thermal gains had.
// **The two coefficients are a property of the individual tyre and cannot be borrowed**, which is a
// measured conclusion rather than a search that came up short — see the fields below.
export struct TyrePressure
{
    // What this tyre is set to cold, **gauge** pascals, and the air temperature it was set at.
    //
    // 28 psi at 20 °C for the Golf: AC's own `PRESSURE_STATIC`, and 20 °C because that is the
    // temperature a pressure is set at rather than a property of the car. A driver who sets his
    // pressures on a cold morning has a different tyre all day, which is a real effect and is what
    // stating the reference rather than assuming it buys.
    double coldPressure = 193053.0;
    double coldReferenceTemperature = 20.0;

    // Where this tyre wants to be, **gauge** pascals — AC's `PRESSURE_IDEAL`, 34 psi for the Golf.
    //
    // **Every pressure-dependent number the car states is quoted at this pressure**, which is what
    // makes it the reference rather than a target: `CornerSetup::tireVerticalRate` and
    // `CornerSetup::rollingResistance` are both the ideal-pressure values, so a tyre sitting exactly
    // here multiplies both by exactly one and the whole model is inert.
    double idealPressure = 234421.0;

    // The cavity's volume, cubic metres. Derived from the tyre's own geometry rather than stated:
    // it is only used for the gas's heat capacity, which is tiny — about 70 J/K against the
    // carcass's thousands — so the gas follows what surrounds it with a time constant of seconds.
    double cavityVolume = 0.0282;

    // W/K from the carcass's inner liner into the air, and from the rim well into it.
    //
    // **Derived so that the two models of this one cavity agree, and that pinned the coefficient the
    // brake model had only bounded.** `WheelThermal::toTyre` states the whole wheel-to-tyre path as
    // 4.0 W/K, of which 2.8 is the air — the liner's 0.6 m² and the rim well's 0.27 m² in series
    // through air stirred by rotation at a bounded 10-30 W/(m²·K). Resolving that series into two
    // conductances requires `h / (1/0.27 + 1/0.6) = 2.8`, so **h = 15.0 W/(m²·K)** — inside the
    // bound, and now a number rather than a range. That gives 9.0 W/K on the liner and 4.05 on the
    // rim, and their series is the 2.8 the brake model already shipped.
    //
    // **Getting this wrong is a double count, and the first draft did.** With the air as a node, the
    // wheel reaches the carcass twice — once through `toTyre`'s lumped 4.0 and again through the gas
    // — unless the lumped path drops its cavity share, which is what `WheelThermal::cavityShare` is
    // for. It was caught by measuring rather than by reading: the air came out five degrees below the
    // carcass on a stint and the rim's grip on it looked too strong.
    double linerConductance = 9.0;
    double rimConductance = 4.05;

    // How rolling resistance follows pressure, as an exponent: `rr = rr_ideal · (P/P_ideal)^-e`.
    //
    // **Bounded, not fitted.** The mechanism is textbook — lower pressure means more sidewall
    // bending and shearing per revolution, so more hysteresis — and published exponents for a
    // pneumatic tyre cluster near 0.5. The one clean fitted figure found, `Cr = 0.1071·P^-0.477`
    // at r² = 0.99, is measured on **bicycle** tyres and is recorded as a neighbour rather than as
    // this tyre's own. AC's `PRESSURE_RR_GAIN 0.55` sits at the top of the 0.4-0.55 band.
    double rollingResistanceExponent = 0.5;

    // What pressure is worth to grip, as the Magic Formula's own two pressure terms on a peak
    // friction coefficient: `1 + linear·dpi + quadratic·dpi²`, with `dpi = (P − P_ideal)/P_ideal`.
    //
    // **Both are exactly 0.0, so the factor is exactly 1.0 at every pressure**, and that is a
    // bit-exact inertness proof rather than an approximate one: a zero times a finite number is a
    // zero, where the vertical-rate and rolling-resistance couplings are computed ratios that only
    // round to one at the ideal.
    //
    // **The form is sourced and the numbers are not, because they are not a property of tyres in
    // general.** The campaign these terms were fitted on — de Hoogh, *Implementing Inflation Pressure
    // and Velocity Effects into the Magic Formula Tyre Model*, TU/e DCT 2005.46, 2005: five passenger
    // car tyres on the TNO Tyre Test Trailer at three pressures each — concludes for the **lateral**
    // peak that the measurements *"do not show a clear tyre independent relation"*, and the two
    // TIME-project reports it cites disagree over whether there is any effect at all. For the
    // **longitudinal** peak an optimum does appear, but fitting this quadratic to that report's own
    // figures at this model's own 4000 N nominal load gives `quadratic` = **−2.25** on a 185/60 R14,
    // **−1.62** on a 205/55 R16 and **+0.47** on a 225/55 R16. **The curvature inverts on the widest
    // tyre in the set**, which is the one nearest this car's 225/40 R18. Marshek & Cuderman's
    // vehicle-level braking tests (SAE 2002-01-0306) reach it from the other end: an optimum on small
    // tyres and *"no noticeable effect at all"* on large ones.
    //
    // **And the closest relative this car has in any published measurement agrees that it is small.**
    // A Goodyear Eagle F1 Asymmetric 255/45 ZR19 on a Flat-Trac at 33, 37 and 41 psi — Singh &
    // Sivaramakrishnan, The Goodyear Tire & Rubber Company, *Extended Pacejka Tire Model for Enhanced
    // Vehicle Stability Control*, arXiv:2305.18422 — moves its peak **lateral** grip by 1.1% across
    // that whole 8 psi at nominal load, monotonically and with no optimum in range, and the authors
    // write that the effect *"doesn't seem to be very dominant"*. **The adaptation they publish for
    // peak grip is then in tread depth and temperature only**, while the cornering-stiffness equation
    // in the same paper does carry explicit pressure terms — so the omission is specific to peak grip.
    //
    // **So this is not a number waiting to be looked up.** Stating those four donor fits on this car
    // at the pressures a driven lap finished at moves its front-to-rear grip balance by **+3.54%,
    // +2.31%, −0.57% and +0.33%** — a change of sign, and an order of magnitude between the smallest
    // and the largest — against the 1.2% of balance the thermal model moved and the seat could feel.
    // **The two nearest this car's tyre are the two smallest.** Choosing
    // one would be choosing this car's handling balance rather than sourcing a coefficient.
    //
    // **There is deliberately no seat knob**, unlike `OSR_TYRE_IDEAL`, which was built to let Dominic
    // choose between two ends of a *sourced* band. Here the literature does not fix the sign, so a
    // seat verdict could only fit a grip law to a feeling. `[.tyre-pressure]` measures what each donor
    // fit would be worth instead, which is the question that can be answered.
    double gripPressureLinear = 0.0;
    double gripPressureQuadratic = 0.0;
};

export struct TyreModel
{
    // The load the coefficients are quoted at. Everything load-dependent is relative to this.
    double nominalLoad = 4000.0;

    // --- lateral ---
    double lateralShape = 1.35;
    // Peak friction coefficient at the nominal load. **Not** the friction the tire will actually
    // produce: that is this, scaled by load sensitivity and again by the surface and by
    // `gripScale`, which is where a thermal or pressure model will multiply in.
    double lateralPeak = 1.05;
    // Cornering stiffness, as a multiple of the nominal load per radian, and the load at which it
    // stops growing. Real cornering stiffness rises with load and then falls away, which the sine
    // of an arctangent describes in one line.
    double lateralStiffness = 20.0;
    double lateralStiffnessLoad = 2.2;
    double lateralCurvature = -1.2;

    // --- longitudinal ---
    double longitudinalShape = 1.65;
    double longitudinalPeak = 1.15;
    double longitudinalStiffness = 20.0;
    double longitudinalCurvature = -1.0;

    // How hard friction falls away with load, as an exponent: mu = peak * (Fz/Fz0)^-sensitivity.
    //
    // A power law rather than the Magic Formula's usual linear term in normalised load, for a
    // reason this game makes routine: linear load sensitivity goes *negative* somewhere above about
    // six times nominal load, and a car landing off a ramp reaches that. A tire with negative
    // friction pushes the car sideways in whichever direction it was already sliding. The power law
    // cannot do that, is non-linear as the brief requires, and is indistinguishable from linear over
    // the range either of them is calibrated in.
    //
    // **One per axis, because a tyre has two and they are not the same number.** This was a single
    // field until 2026-08-23, and the lateral exponent served both — while AC's `tyres.ini`, which
    // is where this car's came from, states `LS_EXP_Y` *and* `LS_EXP_X`. The model was discarding a
    // figure its own data source carries, which is a faithfulness defect whatever it turns out to be
    // worth: the two ends of a stop and a launch sit at opposite ends of the load curve, so an axis
    // that falls off at the wrong rate is wrong in opposite directions at each end.
    //
    // Both default to the same number, which is what makes the split invisible to any car that does
    // not state two — pinned in `TyreLoadSensitivityTests`.
    double lateralLoadSensitivity = 0.15;
    double longitudinalLoadSensitivity = 0.15;

    // Distance constants for the carcass. This is the parameter that makes a tire feel like a tire
    // rather than like a spring-loaded skid: force builds over a distance travelled, not over a
    // time elapsed, so it behaves the same at every speed.
    double lateralRelaxation = 0.50;
    double longitudinalRelaxation = 0.30;

    // Hysteresis in the carcass, N·s/m of deflection rate, and it is load-bearing rather than a
    // refinement. Wheel inertia working against the tire's longitudinal stiffness is a spring-mass
    // oscillator — about 22 Hz for a road wheel — and at a standstill the relaxation term is zero,
    // so nothing damps it at all. A car parked on level ground sat there with its wheels rocking
    // back and forth through a couple of rad/s, throwing 1500 N of longitudinal force fore and aft
    // for ever and pitching the static corner weights out by 2%. A real tire loses that energy as
    // heat in the sidewall. Zero at steady state, where the deflection is not changing, so it costs
    // nothing anywhere else.
    double carcassDamping = 600.0;

    // Pneumatic trail at zero slip, and how quickly it collapses as the patch starts to slide. The
    // trail collapsing before the lateral force peaks is what a driver feels as the wheel going
    // light at the limit — which is the warning, and it arrives before the grip goes.
    //
    // Calibrated so the lateral curve peaks at about 8.6 degrees of slip and the longitudinal at
    // about 11% slip ratio, which is where a road tire does. Placeholder, like everything else.
    //
    // The friction figures are a *road* tire's — a peak around 1.05 lateral — and that is a choice
    // rather than a limitation. A race tire's 1.55 on this placeholder chassis puts the cornering
    // limit above the car's own rollover threshold, which for a 0.52 m centre of gravity on a
    // 1.44 m track is 1.38 g: it would tip before it slid, and every handling case below would be
    // measuring the wrong failure. Sticky tires belong with a low car, and both are data.
    double pneumaticTrail = 0.035;
    double trailFalloff = 8.0;

    // --- the seams the deferred models need ---
    //
    // Runtime scale on friction, kept *outside* the coefficients above so that a thermal, pressure
    // or wear model multiplies here and nothing has to be recalibrated. Baking it into the peak
    // would make every one of those a change to the tire data instead of a change beside it.
    double gripScale = 1.0;
    // Shifts where the peak sits in slip. Wear and temperature both move it; nothing does yet.
    double peakSlipScale = 1.0;

    // The tread's heat balance. Inert unless `VehicleSetup::tyreThermal` switches it on, and inert
    // whatever that says on a tyre that states no grip curve.
    TyreThermal thermal;

    // The air inside it. Inert unless `VehicleSetup::tyrePressure` switches it on, and inert then
    // too on a tyre whose gas is sitting at its own ideal pressure.
    TyrePressure pressure;
};

// What every tyre in this project starts at, degrees Celsius.
//
// **65 is the middle of the plateau this project's curve states**, which is what makes it the right
// default rather than a warm-sounding number: the curve is flat at exactly 1.00 from 55 to 75 °C, so
// a default-constructed state multiplies grip by exactly one and every figure measured before there
// was a thermal model reproduces to the bit. Every performance figure in this project was taken on a
// tyre that is always at its best, and this is the statement of that assumption in one place.
//
// **It was 85 until 2026-08-28 and it moved with the window, not on its own.** The compound's plateau
// slid 20 °C down when the curve stopped being a track tyre's, and this number has to travel with it
// or every fixture in the suite quietly starts measuring an off-plateau tyre — 3 to 4% on the skidpad,
// the 0-100 and the stop, with no physics changed at all. That coupling is asserted rather than
// remembered: `grip.at(tyreDefaultTemperature) == 1.0` is a test.
//
// A fixture that wants a cold tyre says so in its own body — `seedTyreTemperatures` is how.
export inline constexpr double tyreDefaultTemperature = 65.0;

// The carcass's own state: how far the contact patch has been dragged out of line with the wheel,
// and since 2026-08-28 how hot its three layers are.
// This is what makes the model transient rather than instantaneous, and it is per wheel.
export struct TyreState
{
    double longitudinalDeflection = 0.0;
    double lateralDeflection = 0.0;
    // Nothing reads this yet. It is the slot the deferred wear model needs to exist in the
    // serialisable state from the beginning rather than being added to it later.
    double wear = 0.0;

    // Degrees Celsius. The outer skin of the tread, the body of the tread rubber, and the belt and
    // sidewalls behind it. **Grip reads the core** — see `TyreThermal` for the source that says so.
    // The carcass is what carries heat from one corner to the next and makes lap five different from
    // lap one.
    double surfaceTemperature = tyreDefaultTemperature;
    double coreTemperature = tyreDefaultTemperature;
    double carcassTemperature = tyreDefaultTemperature;

    // The air in the cavity, degrees Celsius, and the fourth node since 2026-08-28. Until stage 2 it
    // was folded into the carcass; it is its own node now because **pressure follows this
    // temperature and nothing else**, and the gas is not the carcass — it sits behind the liner and
    // is stirred by the rim as well.
    //
    // **Its seed is not `tyreDefaultTemperature` and must not be.** A default-constructed tyre has
    // to sit at its own *ideal pressure*, the way a default-constructed tread sits on its own
    // plateau, and the gas temperature that does that is a property of the car's cold and ideal
    // pressures rather than of its compound. `seedTyreGasAtIdealPressure` is how, and it is the one
    // seed in this model that is computed rather than stated.
    double gasTemperature = tyreDefaultTemperature;
};

static_assert(std::is_trivially_copyable_v<TyreState>, "per-wheel tire state rides in the vehicle's POD state");

export struct TyreSlip
{
    double slipRatio = 0.0;
    double slipAngle = 0.0;
};

// How fast the carcass is being deformed, which is what the hysteresis term acts on. Reported by
// the relaxation rather than stored, so it cannot go stale against the deflection it belongs to.
export struct TyreDeflectionRate
{
    double longitudinal = 0.0;
    double lateral = 0.0;
};

export struct TyreForces
{
    double longitudinal = 0.0;
    double lateral = 0.0;
    double aligningMoment = 0.0;

    // Dissipation at the contact patch, in watts. Nothing consumes it this milestone; it is
    // computed and exposed because it is precisely the input a thermal model needs, and a model
    // that has to start computing it later has to start plumbing it later too.
    double slipPower = 0.0;

    // How much of the friction ellipse is in use, 0 to about 1. The channel that says whether the
    // car is near the limit, which no single force does.
    double gripUsed = 0.0;

    // Where this tyre's own curves peak, under the load and the compound it was just evaluated at:
    // the slip ratio and the slip angle's tangent at which each direction makes its most force.
    //
    // Exposed for the same reason `slipPower` is — it is computed here anyway, and the alternative
    // is every consumer inventing a constant for it. **A cue that fires at a fixed slip is a cue
    // that means something different on every compound**, and the two that want this are the tyre
    // audio's skid and the pedal rumble, which are the same physical event and must agree. They
    // move with the tyre if they are stated as multiples of these.
    double longitudinalPeakSlip = 0.0;
    double lateralPeakSlip = 0.0;
};
// Where this tyre's aligning moment peaks, in pure lateral slip, at a given load.
//
// **This is the steering limit, stated as a property of the tyre rather than as a number somebody
// drove to.** Aligning moment is the product of lateral force, which is still climbing, and
// pneumatic trail, which is already collapsing — so it turns over *before* grip does, and the slip
// angle where it turns over is the moment the wheel starts going light in a driver's hands. Every
// downstream question about the steering — where a power assist's taper belongs, what the limit cue
// is worth in newton metres — is asked about this point, and until now the answer came from reading
// it off a session trace, which is a seat session per car.
//
// Pure lateral on purpose. On a real lap the front tyre is doing a lot of longitudinal work at the
// same time — measured on Bathurst, the outside front runs |Fx|/Fz between 0.60 and 0.73 through
// the whole slip range — so the *measured* lateral μ there is a component of a nearly saturated
// friction circle rather than the tyre's lateral capability. That is a fact about how the car is
// being driven, not about the tyre, and a per-car constant must not be fitted to it.
export struct TyreAligningPeak
{
    double slipAngle = 0.0;
    double lateralForce = 0.0;
    double aligningMoment = 0.0;
};

// The five entry points. **Defined in Physics/Impl/TyreImpl.cpp, not here, and deliberately**: a
// definition in a module interface is part of that module's BMI, so editing one rebuilt every
// importer of `raceengine` — 107 ninja edges and 95 s for a one-line change to a curve. Declared
// here and defined in an implementation unit, the same edit rebuilds one object file. The reasoning
// each of these carries moved with its body; what is stated here is the contract.
//
// Friction at a given load. Falls with load, which is the single property that makes weight
// transfer cost something and therefore makes every setup change do anything at all.
//
// **The axis is named rather than the peak passed**, and that is the whole of what the exponent
// split cost at the interface. This used to take a peak, which every caller read off the model
// anyway — and once there are two exponents, a caller holding a peak in one hand has to remember
// which exponent goes with it. Naming the axis makes the pair unbreakable: there is no call that
// can put `lateralPeak` against the longitudinal exponent, because neither is spelled at the call.
export [[nodiscard]] double tyreFriction(const TyreModel& model, const TyreAxis axis, const double verticalLoad,
                                         const double surfaceGrip);

// The carcass relaxation, integrated over one tick. Advances `state`'s two deflections and reports
// how fast each moved, which is what the hysteresis term in `evaluateTyre` acts on.
export TyreDeflectionRate relaxTyre(const TyreModel& model, TyreState& state, const double longitudinalVelocity,
                                    const double longitudinalSlipVelocity, const double lateralSlipVelocity,
                                    const double deltaTime);

// The slip the tire is actually working at, read off the deflection rather than off the velocities.
// At steady state this recovers exactly the textbook slip ratio and slip angle; through a transient
// it lags them by the relaxation length, which is the whole point.
export [[nodiscard]] TyreSlip tyreSlip(const TyreModel& model, const TyreState& state);

// Forces from slip and load. `surfaceGrip` is what the contact patch aggregated out of the road it
// is standing on, and multiplies friction without touching the coefficients.
export [[nodiscard]] TyreForces evaluateTyre(const TyreModel& model, const double verticalLoad, const TyreSlip& slip,
                                             const double surfaceGrip, const double longitudinalSlipVelocity = 0.0,
                                             const double lateralSlipVelocity = 0.0,
                                             const TyreDeflectionRate& deflectionRate = {});

// Where this tyre's aligning moment peaks, in pure lateral slip, at a given load. Swept rather than
// solved; a few hundred evaluations, once per car at load time.
export [[nodiscard]] TyreAligningPeak tyreAligningPeak(const TyreModel& model, const double verticalLoad,
                                                       const double surfaceGrip = 1.0);

// --- the thermal model -------------------------------------------------------------------------

// The three nodes' heat capacities and the conductances between them, derived from the tyre's own
// geometry and the tread's material properties. Everything here has a unit; nothing here is fitted.
//
// Recomputed per wheel per tick rather than cached on the setup, which is about twenty flops against
// a tick that casts eighty-four rays — and which means there is no derived copy that can go stale
// against the geometry it came from.
export struct TyreThermalNodes
{
    // J/K.
    double surfaceCapacity = 0.0;
    double coreCapacity = 0.0;
    double carcassCapacity = 0.0;

    // W/K, conduction along the tread's own depth.
    double surfaceToCore = 0.0;
    double coreToCarcass = 0.0;

    // m². The tread band's outer face, which is what convects and what conducts into the road; and
    // both sidewalls, which is what the carcass convects over.
    double treadArea = 0.0;
    double sidewallArea = 0.0;

    // kg, reported because the split between the tread and everything else is the one piece of this
    // derivation worth checking against a published tyre mass by eye.
    double treadMass = 0.0;
    double carcassMass = 0.0;
};

export [[nodiscard]] TyreThermalNodes tyreThermalNodes(const TyreThermal& thermal);

// What one wheel's tread is being asked to absorb and lose over one tick.
//
// **Both generation terms are already computed in this engine and were being thrown away**, which is
// most of the argument that this stage was ready to be built. TRT EVO names them FP, friction power
// at the patch, and SEL, the strain energy loss that *is* rolling resistance — and this model has
// `TyreForces::slipPower` in watts on every wheel every tick and `CornerSetup::rollingResistance`
// beside it. Our SEL is better founded than the published one, which fits `f(F, omega, gamma, P)`
// per tyre from rig tests: rolling resistance is the hysteresis loss by definition, so `Crr·Fz·v` is
// the same energy stated thermodynamically.
export struct TyreThermalInput
{
    // Watts, straight off `TyreForces::slipPower`. Heats the surface, scaled by the one fitted
    // number.
    double slipPower = 0.0;

    // Newtons, and the rolling-resistance coefficient beside it. Their product with the road speed
    // is the strain energy loss, in watts, with no free parameter at all — and it heats the core and
    // the carcass rather than the surface, which is why a tyre warms up on a straight without
    // sliding anywhere.
    double verticalLoad = 0.0;
    double rollingResistance = 0.0;

    // How fast the tread is going over the road, m/s. Sets the residence time in the patch and so
    // the road conduction.
    double roadSpeed = 0.0;
    // How fast the wheel is going through the air, m/s. The two differ by the slip and by nothing
    // else on the ground, and a wheel in the air has one and not the other.
    double airSpeed = 0.0;

    // The contact patch's footprint, metres, or zero for a wheel that is not touching. Derived by
    // the caller from the tyre's own deflection — the chord a radius makes at that penetration — so
    // no constant enters here either.
    double patchLength = 0.0;
    double patchWidth = 0.0;

    // The **rim** this tyre is mounted on, degrees Celsius, and how strongly the carcass is coupled
    // to it, W/K. Bead seats and the inner air together; `WheelHardware::toTyre` in `:Brakes` is
    // where the derivation lives.
    //
    // **It is a plain pair of doubles and not a wheel node, and that is deliberate**: `:Tyre` and
    // `:Brakes` neither import each other and must not start to. The vehicle tick owns both and is
    // the only thing that knows they are bolted together.
    //
    // **Zero conductance is the tyre stage 1 shipped**, which had nothing on the rim side at all,
    // and it leaves the carcass's balance bit for bit what it was.
    double wheelTemperature = 0.0;
    double wheelConductance = 0.0;

    AmbientConditions ambient;
};

// One tick of the heat balance, advancing `state`'s three temperatures.
//
// Each node is integrated **exactly** against the others held at their start-of-tick values, which
// is `relaxTyre`'s treatment and is here for its reason: the closed form is unconditionally stable
// and cannot stretch a time constant with the timestep. All three read the same start-of-tick
// temperatures and are written afterwards, so the answer does not depend on the order they are
// solved in.
export void stepTyreThermal(const TyreThermal& thermal, TyreState& state, const TyreThermalInput& input,
                            const double deltaTime);

// One tick of the cavity air's own balance, advancing `state.gasTemperature`.
//
// **Separate from `stepTyreThermal` on purpose.** The two models switch independently, so the gas
// must not ride inside the tread's stepper and acquire its switch; and the gas is a one-node problem
// with two neighbours, which the same closed form solves exactly.
//
// The air is stirred between the carcass's inner liner and the rim well, and it reads the wheel
// through the same `TyreThermalInput` fields the tread's stepper does — zero conductance when there
// is no brake thermal model, which leaves the gas following the carcass alone.
//
// **Its heat capacity is computed rather than stated**, because it depends on how much air is in
// there: the ideal gas law gives the density from the pressure the tyre is currently at, and about
// 70 J/K comes out. That is small enough that the gas tracks its neighbours with a time constant of
// seconds, which is why it never needed to be a node until pressure did.
export void stepTyreGas(const TyrePressure& pressure, TyreState& state, const TyreThermalInput& input,
                        const double deltaTime);

// What this tyre's temperature is worth to its grip, dimensionless — the curve read at the **core**.
// Exactly 1.0 for a tyre that states no curve, and exactly 1.0 anywhere on the curve's own plateau.
export [[nodiscard]] double tyreTemperatureGrip(const TyreThermal& thermal, const TyreState& state);

// Put a tyre at a stated temperature, all three nodes together. What a fixture calls when it wants a
// cold tyre — a starting temperature is part of a braking or a skidpad measurement the moment there
// is a thermal model, and a fixture that does not say which one it is running is measuring something
// it has not stated.
export void seedTyreTemperature(TyreState& state, const double celsius);

// What this tyre is blowing, **gauge** pascals, at the gas temperature it is currently carrying.
//
// The gas law at constant volume and nothing else: `P_abs = P_cold_abs · T_abs / T_cold_abs`. Exact,
// with no constant to choose and nothing borrowed.
export [[nodiscard]] double tyrePressureAt(const TyrePressure& pressure, const TyreState& state);

// The gas temperature at which this tyre sits at its own ideal pressure, degrees Celsius.
//
// **The inverse of the line above, and it is the model's one computed seed.** Every pressure-dependent
// number a car states — its vertical rate, its rolling resistance — is quoted at the ideal pressure,
// so a tyre seeded here multiplies both by exactly one and the whole suite is inert. For the Golf's
// 28 psi set at 20 °C against an ideal of 34 it comes out at 61.2 °C, which is a fact about AC's own
// two numbers rather than a choice.
export [[nodiscard]] double tyreGasTemperatureAtIdealPressure(const TyrePressure& pressure);

// Put a tyre's gas at the temperature that makes it read its own ideal pressure.
export void seedTyreGasAtIdealPressure(const TyrePressure& pressure, TyreState& state);

// What this tyre's pressure is worth to the carcass's vertical rate, dimensionless.
//
// Linear in **gauge** pressure, which is the constant-belt result: the load is carried by the
// contained air, so the rate follows the pressure that contains it. Exactly 1.0 at the ideal.
export [[nodiscard]] double tyrePressureVerticalRateScale(const TyrePressure& pressure, const TyreState& state);

// And what it is worth to rolling resistance, dimensionless. `(P/P_ideal)^-e`, exactly 1.0 at the
// ideal, and **greater than one below it** — a soft tyre drags.
export [[nodiscard]] double tyrePressureRollingResistanceScale(const TyrePressure& pressure, const TyreState& state);

// And what it is worth to grip, dimensionless. The Magic Formula's `1 + p3·dpi + p4·dpi²` against a
// dimensionless `dpi = (P − P_ideal)/P_ideal`, and **exactly 1.0 at every pressure on the shipped
// car**, because both coefficients are zero and no tyre-independent pair exists to put there.
//
// One factor for both axes, which the sourced form is not: the Magic Formula carries `ppx3/ppx4` and
// `ppy3/ppy4` separately, and only the longitudinal pair has a repeatable shape. That is a second
// reason a stated pair here is a **bound** rather than a model. `TyrePressure` above has the rest.
export [[nodiscard]] double tyrePressureGripScale(const TyrePressure& pressure, const TyreState& state);

// What the tread's temperature and the gas's pressure are worth to grip **together**, dimensionless.
//
// **This is the one place the two multiply**, and it exists so that neither subsystem assigns
// `TyreModel::gripScale` on its own. The tyre is one tyre: a cold over-inflated tyre is not two
// independent corrections that happen to land on the same field, and letting two steppers each write
// it would make the answer depend on which ran last. The pressure factor is exactly 1.0 on the
// shipped car, at every pressure, because its two coefficients are zero.
export [[nodiscard]] double tyreStateGrip(const TyreModel& tyre, const TyreState& state, bool thermal,
                                          bool pressure);

} // namespace raceengine
