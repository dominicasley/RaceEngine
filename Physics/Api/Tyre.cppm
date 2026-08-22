module;

#include <algorithm>
#include <cmath>
#include <type_traits>

export module raceengine.physics:Tyre;

namespace raceengine
{

// The tire, as a Magic Formula with the two things that decide whether a car feels like a car:
// load sensitivity, and relaxation length. Everything else here is arithmetic around them.
//
// Pure and stateless except for the two carcass deflections it is asked to carry, so it can be
// swept, plotted and argued about without a chassis anywhere near it.

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
    double loadSensitivity = 0.15;

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
export [[nodiscard]] double tyreFriction(const TyreModel& model, const double peak, const double verticalLoad,
                                         const double surfaceGrip)
{
    if (verticalLoad <= 0.0)
    {
        return 0.0;
    }

    const auto normalised = std::max(verticalLoad / std::max(model.nominalLoad, 1e-6), 1e-6);

    return peak * std::pow(normalised, -model.loadSensitivity) * model.gripScale * surfaceGrip;
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
export TyreDeflectionRate relaxTyre(const TyreModel& model, TyreState& state, const double longitudinalVelocity,
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
export [[nodiscard]] TyreSlip tyreSlip(const TyreModel& model, const TyreState& state)
{
    return TyreSlip{.slipRatio = state.longitudinalDeflection / std::max(model.longitudinalRelaxation, 1e-6),
                    .slipAngle = std::atan(state.lateralDeflection / std::max(model.lateralRelaxation, 1e-6))};
}

// Forces from slip and load. `surfaceGrip` is what the contact patch aggregated out of the road it
// is standing on, and multiplies friction without touching the coefficients.
export [[nodiscard]] TyreForces evaluateTyre(const TyreModel& model, const double verticalLoad, const TyreSlip& slip,
                                             const double surfaceGrip, const double longitudinalSlipVelocity = 0.0,
                                             const double lateralSlipVelocity = 0.0,
                                             const TyreDeflectionRate& deflectionRate = {})
{
    auto forces = TyreForces{};

    if (verticalLoad <= 0.0)
    {
        return forces;
    }

    const auto lateralFriction = tyreFriction(model, model.lateralPeak, verticalLoad, surfaceGrip);
    const auto longitudinalFriction = tyreFriction(model, model.longitudinalPeak, verticalLoad, surfaceGrip);

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

// Swept rather than solved: `aligningMoment` is a product of a magic formula and a trail falloff and
// has no closed-form turning point worth deriving. A tenth of a degree is far finer than anything
// downstream of it can act on, and the whole sweep is a few hundred evaluations once per car at load
// time.
export [[nodiscard]] TyreAligningPeak tyreAligningPeak(const TyreModel& model, const double verticalLoad,
                                                       const double surfaceGrip = 1.0)
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

} // namespace raceengine
