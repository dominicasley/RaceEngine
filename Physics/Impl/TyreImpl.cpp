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
[[nodiscard]] double tyreFriction(const TyreModel& model, const double peak, const double verticalLoad,
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

} // namespace raceengine
