module;

// Only what the *declarations* below need. <algorithm> and <cmath> moved to TyreImpl.cpp with the
// bodies that use them — see the header of that file for why the bodies are no longer here.
#include <cstdint>
#include <type_traits>

export module raceengine.physics:Tyre;

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
};

// The carcass's own state: how far the contact patch has been dragged out of line with the wheel.
// This is what makes the model transient rather than instantaneous, and it is per wheel.
export struct TyreState
{
    double longitudinalDeflection = 0.0;
    double lateralDeflection = 0.0;
    // Nothing reads this yet. It is the slot the deferred wear model needs to exist in the
    // serialisable state from the beginning rather than being added to it later.
    double wear = 0.0;
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

} // namespace raceengine
