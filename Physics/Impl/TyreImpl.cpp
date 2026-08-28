// The tyre model's bodies. Declarations are in Physics/Api/Tyre.cppm.
//
// This is a **module implementation unit** — `module raceengine.physics;` with no `export` — and
// that is the whole point of the file: an implementation unit produces an object and no BMI, so
// nothing can import it and nothing rebuilds when it changes. A definition left in the interface
// partition is part of the module's BMI instead, and editing one rebuilt every importer of
// `raceengine`: 107 ninja edges, 95 s, for a one-line change to a curve. Here the same edit is one
// object file. Full account and the measurements: docs/build-times.md.
//
// What must NOT move here: anything an importer has to see through the BMI — templates, constexpr
// used at compile time, and in-class member definitions. The tyre model has none of those, which is
// why it went first.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>

module raceengine.physics;

namespace raceengine
{

namespace
{

// The Magic Formula itself: one sine of an arctangent, and the whole shape of a tire curve.
[[nodiscard]] double magicFormula(const double slip, const double peak, const double shape, const double stiffness,
                                  const double curvature)
{
    const auto inner = stiffness * slip;
    const auto shaped = inner - curvature * (inner - std::atan(inner));

    return peak * std::sin(shape * std::atan(shaped));
}

// Where that curve peaks, which the combined-slip normalisation below needs so that the ellipse is
// drawn in units of "fraction of the way to the limit" rather than in raw slip.
[[nodiscard]] double peakSlipOf(const double shape, const double stiffness)
{
    if (stiffness <= 0.0 || shape <= 0.0)
    {
        return 1.0;
    }

    return std::tan(1.5707963267948966 / shape) / stiffness;
}

} // namespace

// Friction at a given load. Falls with load, which is the single property that makes weight
// transfer cost something and therefore makes every setup change do anything at all.
//
// The peak and the exponent are both taken off the axis rather than passed in, which is what stops
// the two exponents from ever being crossed with the wrong peak — see the declaration in Tyre.cppm.
[[nodiscard]] double tyreFriction(const TyreModel& model, const TyreAxis axis, const double verticalLoad,
                                  const double surfaceGrip)
{
    if (verticalLoad <= 0.0)
    {
        return 0.0;
    }

    const auto peak = axis == TyreAxis::Lateral ? model.lateralPeak : model.longitudinalPeak;
    const auto sensitivity =
        axis == TyreAxis::Lateral ? model.lateralLoadSensitivity : model.longitudinalLoadSensitivity;

    const auto normalised = std::max(verticalLoad / std::max(model.nominalLoad, 1e-6), 1e-6);

    return peak * std::pow(normalised, -sensitivity) * model.gripScale * surfaceGrip;
}

// The carcass relaxation, integrated over one tick.
//
// This is the answer to the low-speed singularity, and it is a better one than clamping the slip
// ratio: nothing here divides by wheel speed. What is integrated is the *deflection* of the contact
// patch, which at speed relaxes towards the steady-state slip and at a standstill simply
// accumulates — so a stationary car on a slope is held by a tire acting as a spring, which is what
// a real one does, rather than by a slip calculation that has to be special-cased to avoid dividing
// by zero.
//
// Integrated **exactly** rather than stepped, which is worth one exponential per wheel per tick.
//
// The equation is du/dt = Vs - (|Vx|/sigma)*u, whose solution over a constant tick is a decaying
// exponential towards Vs*sigma/|Vx|. Backward Euler would be unconditionally stable, which is the
// usual reason to reach for it, and would still be wrong here in a way that matters: its decay
// factor is 1/(1+k) where the true one is exp(-k), and at 50 m/s with a half-metre relaxation length
// k is 0.28 and the effective distance constant comes out 14% long. At 100 m/s it is 24% long. The
// relaxation length is a *measured* tire property and the whole point of it is that force builds
// over a fixed distance, so an integrator that quietly stretches it with speed defeats the
// parameter. The closed form has no such error at any speed.
TyreDeflectionRate relaxTyre(const TyreModel& model, TyreState& state, const double longitudinalVelocity,
                             const double longitudinalSlipVelocity, const double lateralSlipVelocity,
                             const double deltaTime)
{
    const auto speed = std::abs(longitudinalVelocity);

    const auto relax = [speed, deltaTime](double& deflection, const double slipVelocity, const double relaxation)
    {
        const auto before = deflection;
        const auto rate = speed / std::max(relaxation, 1e-6);
        const auto decay = rate * deltaTime;

        // At a standstill there is no decay at all and the deflection simply accumulates — the tire
        // becoming a spring, which is what holds a parked car on a slope. Taken as the limit rather
        // than as a special case: the expression below divides by `rate`.
        if (decay < 1e-9)
        {
            deflection += slipVelocity * deltaTime;
            return (deflection - before) / deltaTime;
        }

        const auto settled = slipVelocity / rate;
        deflection = settled + (deflection - settled) * std::exp(-decay);

        return (deflection - before) / deltaTime;
    };

    return TyreDeflectionRate{
        .longitudinal = relax(state.longitudinalDeflection, longitudinalSlipVelocity, model.longitudinalRelaxation),
        .lateral = relax(state.lateralDeflection, lateralSlipVelocity, model.lateralRelaxation)};
}

// The slip the tire is actually working at, read off the deflection rather than off the velocities.
// At steady state this recovers exactly the textbook slip ratio and slip angle; through a transient
// it lags them by the relaxation length, which is the whole point.
[[nodiscard]] TyreSlip tyreSlip(const TyreModel& model, const TyreState& state)
{
    return TyreSlip{.slipRatio = state.longitudinalDeflection / std::max(model.longitudinalRelaxation, 1e-6),
                    .slipAngle = std::atan(state.lateralDeflection / std::max(model.lateralRelaxation, 1e-6))};
}

// Forces from slip and load. `surfaceGrip` is what the contact patch aggregated out of the road it
// is standing on, and multiplies friction without touching the coefficients.
[[nodiscard]] TyreForces evaluateTyre(const TyreModel& model, const double verticalLoad, const TyreSlip& slip,
                                      const double surfaceGrip, const double longitudinalSlipVelocity,
                                      const double lateralSlipVelocity, const TyreDeflectionRate& deflectionRate)
{
    auto forces = TyreForces{};

    if (verticalLoad <= 0.0)
    {
        return forces;
    }

    const auto lateralFriction = tyreFriction(model, TyreAxis::Lateral, verticalLoad, surfaceGrip);
    const auto longitudinalFriction = tyreFriction(model, TyreAxis::Longitudinal, verticalLoad, surfaceGrip);

    const auto lateralLimit = lateralFriction * verticalLoad;
    const auto longitudinalLimit = longitudinalFriction * verticalLoad;

    // Cornering stiffness rises with load and then falls away again, which is what the sine of an
    // arctangent is doing here rather than a straight line.
    const auto corneringStiffness =
        model.lateralStiffness * model.nominalLoad *
        std::sin(2.0 * std::atan(verticalLoad / std::max(model.lateralStiffnessLoad * model.nominalLoad, 1e-6)));
    const auto slipStiffness = model.longitudinalStiffness * verticalLoad;

    // `peakSlipScale` divides the curve's stiffness, which is what actually moves the peak along
    // the slip axis while leaving its height alone. Scaling the normalisation below instead —
    // which was the first attempt — does nothing whatever in pure slip, because the same factor
    // multiplies back out one line later. An inert hook is worse than no hook: it reads as a
    // feature and behaves as a comment.
    const auto scale = std::max(model.peakSlipScale, 1e-6);
    const auto lateralB = lateralLimit > 1e-9 ? corneringStiffness / (model.lateralShape * lateralLimit * scale) : 0.0;
    const auto longitudinalB =
        longitudinalLimit > 1e-9 ? slipStiffness / (model.longitudinalShape * longitudinalLimit * scale) : 0.0;

    const auto lateralPeakSlip = peakSlipOf(model.lateralShape, lateralB);
    const auto longitudinalPeakSlip = peakSlipOf(model.longitudinalShape, longitudinalB);

    // Combined slip, as a friction ellipse: the two slips are normalised by where their own curves
    // peak, combined as a vector, and the resulting force is shared out along that vector. Full
    // braking and full cornering are then not simultaneously available, because the vector reaches
    // the limit sooner than either component does.
    forces.longitudinalPeakSlip = longitudinalPeakSlip;
    forces.lateralPeakSlip = lateralPeakSlip;

    const auto normalisedRatio = slip.slipRatio / std::max(longitudinalPeakSlip, 1e-9);
    const auto normalisedAngle = std::tan(slip.slipAngle) / std::max(lateralPeakSlip, 1e-9);
    const auto combined = std::hypot(normalisedRatio, normalisedAngle);

    if (combined < 1e-12)
    {
        return forces;
    }

    const auto alongRatio = normalisedRatio / combined;
    const auto alongAngle = normalisedAngle / combined;

    forces.longitudinal =
        alongRatio * magicFormula(combined * longitudinalPeakSlip, longitudinalLimit, model.longitudinalShape,
                                  longitudinalB, model.longitudinalCurvature);
    forces.lateral = -alongAngle * magicFormula(combined * lateralPeakSlip, lateralLimit, model.lateralShape, lateralB,
                                                model.lateralCurvature);

    // Pneumatic trail, collapsing as the patch starts to slide — which is why the wheel goes light
    // in a driver's hands *before* the tire lets go, and the most useful thing a tire tells them.
    // Carcass hysteresis: the tread is a Kelvin-Voigt element, so its force is `k*u + c*du/dt` and
    // not `k*u` alone. It vanishes at steady state, where the deflection is not changing, and only
    // ever damps a transient — including the standstill one that has nothing else to damp it.
    //
    // **The two axes take opposite signs, and that is not a typo.** Longitudinal force runs *with*
    // its deflection (more deflection, more forward force) and lateral force runs *against* its own
    // (positive slip angle, negative lateral force), so writing the same sign twice damps one axis
    // and drives the other. Written with a minus on both, the wheel-spin feedback came out
    // positive: torque grew with wheel speed and the wheel reached 2800 rad/s in six seconds.
    forces.longitudinal = std::clamp(forces.longitudinal + model.carcassDamping * deflectionRate.longitudinal,
                                     -longitudinalLimit, longitudinalLimit);
    forces.lateral =
        std::clamp(forces.lateral - model.carcassDamping * deflectionRate.lateral, -lateralLimit, lateralLimit);

    const auto trail = model.pneumaticTrail / (1.0 + model.trailFalloff * std::abs(std::tan(slip.slipAngle)));

    // **The textbook's -t*Fy, and the minus is the cross product, not a convention to argue with.**
    // The trail means the lateral resultant acts *behind* the patch centre, so the couple about the
    // patch normal is (-t z) x (Fy x) = -t*Fy about y — the lever's minus survives whatever sign Fy
    // itself carries. This line used to read `trail * forces.lateral`, reasoned from "the lateral
    // force is already signed opposite the slip angle" — a chain that skipped the lever — and the
    // inversion was invisible everywhere it was looked for: it is ~2% of the chassis yaw moment, so
    // every handling criterion passed, and the criterion-10 rack check takes an absolute value. Where
    // it was 100% of the answer was the driver's hands: at the rack it opposed the mechanical trail
    // instead of joining it, leaving stage one slightly *pro-steer* at speed. Measured on the Golf at
    // 15 m/s under +0.1 demand: the mechanical-trail term was 34 N·m a corner restoring and this term
    // 31-40 N·m against it, for a rim torque of +0.6 N·m *into* the lock. With the lever's minus the
    // same run reports the rim torque opposing the lock, which is what a front tyre has ever done.
    forces.aligningMoment = -trail * forces.lateral;

    forces.slipPower =
        std::abs(forces.longitudinal * longitudinalSlipVelocity) + std::abs(forces.lateral * lateralSlipVelocity);

    forces.gripUsed = combined;

    return forces;
}

// Swept rather than solved: `aligningMoment` is a product of a magic formula and a trail falloff and
// has no closed-form turning point worth deriving. A tenth of a degree is far finer than anything
// downstream of it can act on, and the whole sweep is a few hundred evaluations once per car at load
// time.
[[nodiscard]] TyreAligningPeak tyreAligningPeak(const TyreModel& model, const double verticalLoad,
                                                const double surfaceGrip)
{
    auto found = TyreAligningPeak{};

    if (verticalLoad <= 0.0)
    {
        return found;
    }

    constexpr auto step = 0.1 * 0.017453292519943295;
    constexpr auto limit = 25.0 * 0.017453292519943295;

    for (auto angle = step; angle <= limit; angle += step)
    {
        const auto forces =
            evaluateTyre(model, verticalLoad, TyreSlip{.slipRatio = 0.0, .slipAngle = angle}, surfaceGrip);

        if (std::abs(forces.aligningMoment) > std::abs(found.aligningMoment))
        {
            found = TyreAligningPeak{
                .slipAngle = angle, .lateralForce = forces.lateral, .aligningMoment = forces.aligningMoment};
        }
    }

    return found;
}

// --- the thermal model -------------------------------------------------------------------------

// The nodes, out of the tyre's own size and the tread's own material. Twenty flops, recomputed
// rather than cached, so nothing here can go stale against the geometry it came from.
[[nodiscard]] TyreThermalNodes tyreThermalNodes(const TyreThermal& thermal)
{
    auto nodes = TyreThermalNodes{};

    constexpr auto pi = 3.141592653589793;

    const auto solid = std::clamp(1.0 - thermal.voidFraction, 1e-3, 1.0);
    const auto treadThickness = std::max(thermal.grooveDepth + thermal.underTread, 1e-4);
    const auto surfaceThickness = std::clamp(thermal.surfaceThickness, 1e-5, 0.5 * treadThickness);

    nodes.treadArea = 2.0 * pi * std::max(thermal.outerRadius, 1e-3) * std::max(thermal.treadWidth, 1e-3);
    nodes.sidewallArea =
        2.0 * pi * std::max(thermal.outerRadius * thermal.outerRadius - thermal.rimRadius * thermal.rimRadius, 0.0);

    // The grooved band carries rubber over only `solid` of its area; the undertread below the groove
    // floor is continuous and carries all of it.
    const auto rubberVolume = nodes.treadArea * (thermal.grooveDepth * solid + thermal.underTread);
    const auto surfaceVolume = nodes.treadArea * surfaceThickness * solid;
    const auto coreVolume = std::max(rubberVolume - surfaceVolume, 1e-9);

    nodes.treadMass = (surfaceVolume + coreVolume) * thermal.density;

    // **What is left of a published tyre mass is the carcass**, which is why the mass is stated on
    // the model at all: belt, plies, liner, sidewalls and beads are a construction nobody publishes
    // the dimensions of, and their total is a number every manufacturer does publish. The floor is a
    // guard against a mass so small it would say the tread is the whole tyre.
    nodes.carcassMass = std::max(thermal.tyreMass - nodes.treadMass, 0.1 * nodes.treadMass);

    nodes.surfaceCapacity = surfaceVolume * thermal.density * thermal.specificHeat;
    nodes.coreCapacity = coreVolume * thermal.density * thermal.specificHeat;

    // The carcass takes the tread's specific heat, which over-states it: a belt package is perhaps a
    // fifth steel by mass, and steel's is a quarter of rubber's. Stated rather than corrected,
    // because splitting a published tyre mass into rubber and steel needs a construction nobody
    // publishes — and the carcass is the slowest node, so what it moves is the length of a stint
    // rather than the shape of a lap.
    nodes.carcassCapacity = nodes.carcassMass * thermal.specificHeat;

    // Fourier along the tread's depth, between the nodes' own centroids. The conduction area through
    // the grooved band is the rubber's, not the band's.
    const auto surfaceDepth = 0.5 * surfaceThickness;
    const auto coreDepth = surfaceThickness + 0.5 * (treadThickness - surfaceThickness);

    nodes.surfaceToCore = thermal.conductivity * nodes.treadArea * solid / std::max(coreDepth - surfaceDepth, 1e-6);
    nodes.coreToCarcass =
        thermal.conductivity * nodes.treadArea / std::max(treadThickness - coreDepth + thermal.carcassHalfDepth, 1e-6);

    return nodes;
}

namespace
{

// Zero Celsius in kelvin, and the Stefan-Boltzmann constant — here rather than beside the gas
// constants below because the radiation term needs them first. One anonymous namespace per
// translation unit, so the gas-law block further down shares these.
constexpr auto absoluteZero = 273.15;
constexpr auto stefanBoltzmann = 5.670374419e-8;

// Grey-body radiation from an exterior surface to surroundings at the air's temperature, as a
// conductance in W/K — the exact secant of the fourth power between the two temperatures,
// `eps·sigma·A·(Ts² + Ta²)(Ts + Ta)`, which is the brake disc's own convention: the full T⁴
// difference, linearised only in how far the temperature moves within one tick. Zero emissivity is
// exactly zero conductance, and adding an exact 0.0 to a conductance sum is the identity — which is
// what keeps a car that states no emissivity bit-identical to the model before the term existed.
[[nodiscard]] double radiativeConductance(const double emissivity, const double area, const double celsius,
                                          const double airCelsius)
{
    if (emissivity <= 0.0)
    {
        return 0.0;
    }

    const auto surface = celsius + absoluteZero;
    const auto ambient = airCelsius + absoluteZero;

    return std::clamp(emissivity, 0.0, 1.0) * stefanBoltzmann * std::max(area, 0.0) *
           (surface * surface + ambient * ambient) * (surface + ambient);
}

// Forced convection from a cylinder in cross flow — the Hilpert correlation, which is what the
// thermoRIDE model (Farroni et al., *Appl. Sci.* 2020, 10, 1604) states for a tyre:
//
//     h = (k_air / L) · 0.0239 · (V · L / nu)^m
//
// **The constant and the exponent reconcile with the textbook, and that is worth recording.** The
// standard Hilpert table for Reynolds numbers of 4e4 to 4e5 gives `Nu = 0.027 · Re^0.805 · Pr^(1/3)`,
// and air's Prandtl number is 0.707, whose cube root is 0.891: 0.027 × 0.891 = 0.0241, against the
// paper's 0.0239. So the paper has folded air's Prandtl term into its constant, which confirms both
// the constant that survived text extraction and the exponent that did not — the 0.805 used here is
// the textbook's and the code says so rather than claiming to have read it off the paper.
//
// A road wheel at 100 km/h sits at Re ≈ 1.2e6, which is past the correlation's stated band, so this
// is an extrapolation of one decade. Stated rather than hidden; the alternative is a correlation
// nobody published for the case a car actually spends its time in.
[[nodiscard]] double forcedConvection(const double speed, const double diameter, const double filmTemperature)
{
    const auto conductivity = airConductivity(filmTemperature);
    const auto viscosity = airKinematicViscosity(filmTemperature);

    if (diameter <= 0.0 || viscosity <= 0.0)
    {
        return 0.0;
    }

    const auto reynolds = std::abs(speed) * diameter / viscosity;
    if (reynolds <= 0.0)
    {
        return 0.0;
    }

    return conductivity / diameter * 0.0239 * std::pow(reynolds, 0.805);
}

// One node, integrated exactly against the others held where they were at the start of the tick.
// `conductance` is the sum of the paths leaving it and `weighted` the same paths times the
// temperatures at their far ends, so the equilibrium this node is falling towards is their ratio.
[[nodiscard]] double advanceNode(const double temperature, const double capacity, const double generation,
                                 const double conductance, const double weighted, const double deltaTime)
{
    if (capacity <= 0.0)
    {
        return temperature;
    }

    if (conductance <= 1e-12)
    {
        return temperature + generation * deltaTime / capacity;
    }

    const auto equilibrium = (generation + weighted) / conductance;

    return equilibrium + (temperature - equilibrium) * std::exp(-conductance * deltaTime / capacity);
}

} // namespace

void stepTyreThermal(const TyreThermal& thermal, TyreState& state, const TyreThermalInput& input,
                     const double deltaTime)
{
    if (deltaTime <= 0.0)
    {
        return;
    }

    const auto nodes = tyreThermalNodes(thermal);
    if (nodes.surfaceCapacity <= 0.0 || nodes.coreCapacity <= 0.0 || nodes.carcassCapacity <= 0.0)
    {
        return;
    }

    const auto air = input.ambient.airTemperature;
    const auto road = input.ambient.trackTemperature;

    // --- what goes in ---
    //
    // FP heats the skin, because sliding happens at the skin. SEL is hysteresis through the whole
    // rubber, so it is shared between the core and the carcass **by their heat capacities**, which
    // is a share derived from the geometry rather than a second fitted number.
    const auto frictionPower = std::max(input.slipPower, 0.0) * std::max(thermal.frictionToTread, 0.0);
    const auto rollingPower = std::max(input.rollingResistance * input.verticalLoad * std::abs(input.roadSpeed), 0.0);
    const auto coreShare = nodes.coreCapacity / (nodes.coreCapacity + nodes.carcassCapacity);

    // --- what comes out ---
    const auto diameter = 2.0 * std::max(thermal.outerRadius, 1e-3);

    const auto surfaceAir = std::max(forcedConvection(input.airSpeed, diameter, 0.5 * (state.surfaceTemperature + air)),
                                     thermal.naturalConvection) *
                            nodes.treadArea;
    const auto carcassAir = std::max(forcedConvection(input.airSpeed, diameter, 0.5 * (state.carcassTemperature + air)),
                                     thermal.naturalConvection) *
                            nodes.sidewallArea;

    // Conduction into the road, and there is no fitted coefficient in it either. A tread element
    // entering the patch meets the road as a step, so the heat it loses while it is in there is the
    // semi-infinite solution averaged over the residence time: `h = 2·e / sqrt(pi·t)`, where `e` is
    // the tread rubber's own thermal effusivity `sqrt(k·rho·c)` — the same sourced trio the capacity
    // came from. Two bodies in contact settle at an effusivity-weighted mean rather than at either
    // one's temperature, so only `e_road / (e_tread + e_road)` of the difference is felt.
    //
    // Multiplied by the **patch** area rather than the tread's, which is exactly the duty cycle: an
    // element is in contact for that fraction of each revolution.
    //
    // The residence time is clamped at both ends. Below a millisecond the semi-infinite solution is
    // describing a skin thinner than the surface node, and above a second the element is not really
    // seeing a step any more — a parked car's tread and the road under it are equalising, which this
    // term then reports as a slow conduction and is the right answer for the wrong reason.
    auto roadConductance = 0.0;
    const auto patchArea = std::max(input.patchLength, 0.0) * std::max(input.patchWidth, 0.0);
    if (patchArea > 0.0)
    {
        const auto residence = std::clamp(input.patchLength / std::max(std::abs(input.roadSpeed), 1e-3), 1e-3, 1.0);
        const auto effusivity = std::sqrt(thermal.conductivity * thermal.density * thermal.specificHeat);
        const auto contact = 2.0 * effusivity / std::sqrt(3.141592653589793 * residence) * thermal.roadEffusivity /
                             std::max(effusivity + thermal.roadEffusivity, 1e-6);

        // And the interface itself, in series, because the expression above assumes the two surfaces
        // touch everywhere and rubber on rough asphalt does not. **Unstated is perfect contact and is
        // the expression above unchanged**, which is what keeps this addition inert to the bit.
        const auto interfaceConductance = thermal.roadContactConductance;
        const auto resisted =
            interfaceConductance > 0.0 && contact > 0.0 ? 1.0 / (1.0 / contact + 1.0 / interfaceConductance) : contact;

        // Only the rubber conducts: `patchArea` is the geometric chord times the section width and
        // the grooves are part of it, so a stated fraction takes them out. **The groove is a hole in
        // the conducting area and not a second path in parallel** — still air across a 7.5 mm groove
        // is about 3.5 W/(m²·K), a thousandth of the rubber's path, so what the groove carries is
        // arithmetic noise and modelling it as nothing is exact to that part. It scales the whole
        // series composition above, since both the semi-infinite term and the interface act only
        // over the area that touches. **One is the gross patch and is the expression above
        // unchanged**, which is what keeps this addition inert to the bit too.
        const auto conductingArea =
            thermal.roadAreaFraction < 1.0 ? patchArea * std::clamp(thermal.roadAreaFraction, 0.0, 1.0) : patchArea;

        roadConductance = resisted * conductingArea;
    }

    // **And the same exterior surfaces radiate**, grey-body against surroundings taken at the air's
    // temperature — the tread band from the surface node, both sidewalls from the carcass, no area
    // invented that the convection above does not already use. At a standstill this is the larger
    // half of the air-side cooling (about 7 W/(m²·K) at 90 °C against the still-air floor's 5); at
    // speed it is 6-10% of the forced path. The unstated case is an exact 0.0 added to each sum,
    // which is the identity — a car with no emissivity is the pre-radiation model to the bit. A
    // sky-temperature refinement is deliberately out of scope, as it is for the disc and the wheel.
    const auto surfaceRadiation =
        radiativeConductance(thermal.emissivity, nodes.treadArea, state.surfaceTemperature, air);
    const auto carcassRadiation =
        radiativeConductance(thermal.emissivity, nodes.sidewallArea, state.carcassTemperature, air);

    // --- and the three nodes, all reading the same start-of-tick temperatures ---
    const auto surface = state.surfaceTemperature;
    const auto core = state.coreTemperature;
    const auto carcass = state.carcassTemperature;

    state.surfaceTemperature = advanceNode(
        surface, nodes.surfaceCapacity, frictionPower,
        surfaceAir + surfaceRadiation + roadConductance + nodes.surfaceToCore,
        (surfaceAir + surfaceRadiation) * air + roadConductance * road + nodes.surfaceToCore * core, deltaTime);

    state.coreTemperature =
        advanceNode(core, nodes.coreCapacity, rollingPower * coreShare, nodes.surfaceToCore + nodes.coreToCarcass,
                    nodes.surfaceToCore * surface + nodes.coreToCarcass * carcass, deltaTime);

    // **The carcass is where brake heat arrives**, through the bead seats and the inner air, and it
    // arrives nowhere else: a rim cannot reach the tread except through the body of the tyre. That
    // is most of why the path is worth so little — the carcass loses to the air over both sidewalls
    // before the core sees any of it. Zero conductance is stage 1's carcass, expression for
    // expression, which is what keeps the tyre model inert to this addition.
    const auto rim = std::max(input.wheelConductance, 0.0);

    state.carcassTemperature =
        rim > 0.0 ? advanceNode(carcass, nodes.carcassCapacity, rollingPower * (1.0 - coreShare),
                                nodes.coreToCarcass + carcassAir + carcassRadiation + rim,
                                nodes.coreToCarcass * core + (carcassAir + carcassRadiation) * air +
                                    rim * input.wheelTemperature,
                                deltaTime)
                  : advanceNode(carcass, nodes.carcassCapacity, rollingPower * (1.0 - coreShare),
                                nodes.coreToCarcass + carcassAir + carcassRadiation,
                                nodes.coreToCarcass * core + (carcassAir + carcassRadiation) * air, deltaTime);
}

[[nodiscard]] double tyreTemperatureGrip(const TyreThermal& thermal, const TyreState& state)
{
    return thermal.grip.at(state.coreTemperature);
}

void seedTyreTemperature(TyreState& state, const double celsius)
{
    state.surfaceTemperature = celsius;
    state.coreTemperature = celsius;
    state.carcassTemperature = celsius;
    state.gasTemperature = celsius;
}

// --- the air inside, and what its pressure is worth ---

namespace
{

// Dry air's gas constant and constant-volume specific heat. (`absoluteZero` lives with the thermal
// constants above — same anonymous namespace.)
//
// These are the only new material constants stage 2 needs and both are textbook: 287.05
// J/(kg·K) for the specific gas constant of dry air, and 718 J/(kg·K) for c_v. **Nothing about the
// gas is fitted** — its mass comes from the pressure it is at and the volume it is in.
constexpr auto airGasConstant = 287.05;
constexpr auto airSpecificHeatConstantVolume = 718.0;

} // namespace

[[nodiscard]] double tyrePressureAt(const TyrePressure& pressure, const TyreState& state)
{
    const auto coldKelvin = pressure.coldReferenceTemperature + absoluteZero;
    if (coldKelvin <= 0.0)
    {
        return pressure.coldPressure;
    }

    const auto gasKelvin = std::max(state.gasTemperature + absoluteZero, 0.0);

    return (pressure.coldPressure + atmosphericPressure) * gasKelvin / coldKelvin - atmosphericPressure;
}

[[nodiscard]] double tyreGasTemperatureAtIdealPressure(const TyrePressure& pressure)
{
    const auto coldAbsolute = pressure.coldPressure + atmosphericPressure;
    if (coldAbsolute <= 0.0)
    {
        return pressure.coldReferenceTemperature;
    }

    const auto coldKelvin = pressure.coldReferenceTemperature + absoluteZero;

    return coldKelvin * (pressure.idealPressure + atmosphericPressure) / coldAbsolute - absoluteZero;
}

void seedTyreGasAtIdealPressure(const TyrePressure& pressure, TyreState& state)
{
    state.gasTemperature = tyreGasTemperatureAtIdealPressure(pressure);
}

[[nodiscard]] double tyrePressureVerticalRateScale(const TyrePressure& pressure, const TyreState& state)
{
    if (pressure.idealPressure <= 0.0)
    {
        return 1.0;
    }

    // Gauge pressure, because it is the difference across the carcass that carries the load. Floored
    // at zero rather than allowed negative: a tyre below atmospheric is not a tyre this model has
    // anything to say about, and a negative rate would be a spring pulling the car down.
    return std::max(tyrePressureAt(pressure, state), 0.0) / pressure.idealPressure;
}

[[nodiscard]] double tyrePressureRollingResistanceScale(const TyrePressure& pressure, const TyreState& state)
{
    if (pressure.idealPressure <= 0.0)
    {
        return 1.0;
    }

    // Floored well above zero, because the power law diverges at zero pressure and a flat tyre's
    // rolling resistance is a different physical problem rather than a very large number.
    const auto ratio = std::max(tyrePressureAt(pressure, state) / pressure.idealPressure, 0.1);

    return std::pow(ratio, -pressure.rollingResistanceExponent);
}

[[nodiscard]] double tyrePressureGripScale(const TyrePressure& pressure, const TyreState& state)
{
    if (pressure.idealPressure <= 0.0)
    {
        return 1.0;
    }

    // The Magic Formula's own dimensionless pressure increment, referred to the pressure every
    // number this tyre states is quoted at.
    const auto dpi = (tyrePressureAt(pressure, state) - pressure.idealPressure) / pressure.idealPressure;

    // **Exactly 1.0 whenever both coefficients are zero, at any pressure**, because a zero times a
    // finite number is a zero. That is the shipped car, and it is the one coupling here whose
    // inertness is bit-exact rather than a part in a million.
    const auto scale = 1.0 + pressure.gripPressureLinear * dpi + pressure.gripPressureQuadratic * dpi * dpi;

    // Floored, because a stated pair is somebody's fit over the three pressures it was measured at
    // and a downward parabola extrapolated far enough reaches zero grip, which is not a tyre. The
    // floor cannot round the shipped car: `std::max(1.0, 0.1)` is 1.0.
    return std::max(scale, 0.1);
}

[[nodiscard]] double tyreStateGrip(const TyreModel& tyre, const TyreState& state, const bool thermal,
                                   const bool pressure)
{
    // **Multiplied, and in this one place.** Both factors are exactly 1.0 when their switch is off,
    // so a car with neither model on comes out of here bit for bit unchanged — which is what lets
    // the caller multiply unconditionally instead of branching.
    auto scale = 1.0;

    if (thermal)
    {
        scale *= tyreTemperatureGrip(tyre.thermal, state);
    }

    if (pressure)
    {
        scale *= tyrePressureGripScale(tyre.pressure, state);
    }

    return scale;
}

void stepTyreGas(const TyrePressure& pressure, TyreState& state, const TyreThermalInput& input,
                 const double deltaTime)
{
    // The mass of air in there, from the pressure it is currently at. About 98 grams for a 225/40 R18
    // at 28 psi, which is 70 J/K — three orders below the carcass, and the reason this node could be
    // folded into the carcass right up until something needed to read its temperature.
    const auto absolute = std::max(tyrePressureAt(pressure, state) + atmosphericPressure, 1.0);
    const auto kelvin = std::max(state.gasTemperature + absoluteZero, 1.0);
    const auto mass = absolute * pressure.cavityVolume / (airGasConstant * kelvin);
    const auto capacity = mass * airSpecificHeatConstantVolume;

    // Two neighbours: the carcass through the liner, and the wheel through the rim well. The second
    // is zero unless there is a brake thermal model to supply a rim temperature, and zero leaves the
    // gas following the carcass alone — which is exactly what it did when it was part of it.
    const auto liner = std::max(pressure.linerConductance, 0.0);
    const auto rim = input.wheelConductance > 0.0 ? std::max(pressure.rimConductance, 0.0) : 0.0;

    state.gasTemperature = advanceNode(state.gasTemperature, capacity, 0.0, liner + rim,
                                       liner * state.carcassTemperature + rim * input.wheelTemperature, deltaTime);
}

} // namespace raceengine
