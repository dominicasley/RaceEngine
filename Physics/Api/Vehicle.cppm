module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.physics:Vehicle;

import raceengine.assists;

import :Ambient;
import :Brakes;
import :Contact;
import :ContactPatch;
import :PhysicsWorld;
import :ProvingGround;
import :RigidBody;
import :Suspension;
import :Telemetry;
import :Tyre;

namespace raceengine
{

// The car, assembled: a chassis that is one rigid body, four corners that each carry one degree of
// freedom, and the forces between them. The tick is a pure function of (setup, state, input, dt) —
// no clock, no globals, no unseeded randomness — which costs nothing now and is the whole of what
// makes a later netcode retrofit possible instead of a rewrite.
//
// **Every number below is a placeholder** for a mid-size car unless a real one has been supplied,
// and is marked where it is stated. They are documented real-world values rather than invented
// ones, so the car behaves plausibly while the model is being validated; replacing them is a data
// change and not a code change.

// A piecewise-linear curve, which is the damper interface the brief asks for rather than a scalar.
// Low and high speed knees are simply more points; a linear damper is two. Nothing here needs to
// change to go from one to the other, which is the point of taking a curve now.
export struct Curve
{
    // Ascending in x. Outside the range the end slope is held rather than extrapolated, because a
    // damper asked about a velocity past its last measured point should be stiff, not wild.
    std::vector<glm::dvec2> points;

    [[nodiscard]] double at(const double x) const
    {
        if (points.empty())
        {
            return 0.0;
        }

        if (points.size() == 1 || x <= points.front().x)
        {
            return points.front().y;
        }

        if (x >= points.back().x)
        {
            return points.back().y;
        }

        for (auto index = std::size_t{1}; index < points.size(); index++)
        {
            if (x <= points[index].x)
            {
                const auto& low = points[index - 1];
                const auto& high = points[index];
                const auto span = high.x - low.x;

                return span > 0.0 ? low.y + (high.y - low.y) * (x - low.x) / span : low.y;
            }
        }

        return points.back().y;
    }
};

// A linear damper stated as a curve, so the common case is not a special case. Bump and rebound
// rates differ on every real damper and are separate arguments here for that reason.
export [[nodiscard]] Curve linearDamper(const double bumpRate, const double reboundRate);

// A wheel-referred kneed damper carried onto the shaft, which is AC's statement turned into this
// model's: `rate` up to `knee` metres per second of *wheel* velocity and `fastRate` above it. The
// shaft moves `ratio` times as fast as the wheel while carrying `1/ratio` of its force, so a
// wheel-referred coefficient divides by the ratio squared and the knee's speed multiplies by it —
// both halves, or the knee lands at the wrong speed. The ratio is the **damper's** and the
// signature says so: a spring's ratio converts a spring rate and nothing else, which is why the
// cross-overload does not exist to call.
export [[nodiscard]] Curve kneedDamper(const double bumpRate, const double fastBumpRate, const double bumpKnee,
                                       const double reboundRate, const double fastReboundRate, const double reboundKnee,
                                       const DamperKinematics& kinematics);
Curve kneedDamper(double, double, double, double, double, double, const SpringKinematics&) = delete;

// How many Maxwell elements a stop's dynamic branch carries. **Five, because that is the number
// the source needed** — Pech et al. approximate their specimen's dynamic-stiffening envelope by a
// piece-wise linear function and read the bend points off as contact points; five segments fit it
// (VSD 2024, 10.1080/00423114.2024.2378858, Figure 16). It is a fixed array rather than a vector
// because `CornerState` carries one of these per corner and that struct is memcpy'd by the
// harness.
export inline constexpr std::size_t jounceElementCount = 5;

// One Maxwell element of a jounce bumper's dynamic branch: a spring in series with a damper, with
// its own free travel before it touches anything.
//
// **This is where a real bump stop's rate dependence lives, and its absence is what flipped the
// car** (docs/suspension-fidelity-brief.md, 2026-08-30). A Maxwell element is a pure spring well
// above its cut-off frequency, transmits nothing well below it, and dissipates most where the
// excitation sits on it — which is exactly "stiff and lossy at strike rates, transparent
// quasi-statically". A viscous constant cannot be any of those things: it is proportional to
// velocity everywhere, which is why the shipped 40000 N·s/m has to be five times a corner's
// critical damping to stop a pogo.
export struct JounceElement
{
    // How far past the stop's own touch point this element starts carrying, metres. The source's
    // "free travel"; the elements differ in it, which is what makes the branch progressive.
    double contactPoint = 0.0;

    // N/m, the series spring — and therefore the force this element carries per metre at
    // frequencies well above its cut-off.
    double stiffness = 0.0;

    // rad/s, `stiffness / dampingCoefficient`. The source parameterises the pair this way round on
    // purpose: the cut-off is what the fit is conditioned on, and it is the quantity a stability
    // bound is written against.
    double cutoff = 0.0;
};

// Everything one stop remembers between ticks. Zero is "not touching anything", which is the state
// a stop is reset to the moment it leaves contact — the source resets its elements after each load
// event, and a friction element that remembers a stroke it is no longer in is remembering the
// wrong stroke.
export struct TravelStopState
{
    // The Dahl element's output, normalised to ±1 so the saturation stays `TravelStop::hysteresis`
    // times the elastic force and the field means the same thing whichever branch is running.
    double dahl = 0.0;

    std::array<double, jounceElementCount> elementForce{};
    std::array<double, jounceElementCount> elementInput{};
};

static_assert(std::is_trivially_copyable_v<TravelStopState>, "CornerState carries one and is copied by bytes");

// A stop's force split the way the corner integrates it (2026-09-08, docs/stop-element-brief.md):
// the part known before the step — the elastic law, the hysteresis, the sourced branch — and the
// viscous coefficient the step solves. `explicitForce` rides the generalised force as every
// element's force does; `viscousCoefficient` joins the damper's slope in the corner's implicit
// divisor, so the viscous share is the coefficient times the velocity the step *arrives at* and
// never the one it started with. Both are zero out of contact.
export struct TravelStopTerms
{
    double explicitForce = 0.0;
    double viscousCoefficient = 0.0;
};

// A stop that comes in gradually and then very hard. `gap` is how much travel there is before it
// touches; past that the force rises as the deflection to `progression`, so it is soft where it
// first bites and immovable at the end. A linear stop either lets the suspension through it or
// hammers the chassis the instant it touches.
export struct TravelStop
{
    double gap = 0.05;
    double rate = 250000.0;
    double progression = 2.0;

    // Hysteresis, and it is not decoration. A stop modelled as a pure spring returns almost all of
    // what an impact puts into it, so a car dropped onto its stops pogos instead of settling — which
    // is exactly what this model did before the term existed, bouncing higher than it was dropped
    // from. Real bump stops are microcellular elastomer and lose a great deal of it as heat.
    //
    // **"A great deal" is wrong and this constant is not how the material loses it** (2026-08-29).
    // BASF's own Cellasto technical brochure characterises the material's dissipation as the
    // hysteresis-loop area over the deformation work, "approximately between 10 % and 20 % (static)"
    // — "relatively low for Cellasto (low damping) and is one of the reasons for its extreme fatigue
    // strength" — not as a viscous constant at all. The `hysteresis` field below is that statement's
    // shape; this viscous constant survives because it is what every current car ships with and what
    // the seat has accepted, and zeroing it is a decision for the seat, not for a source.
    //
    // **What the number means since 2026-09-08** (docs/stop-element-brief.md): the viscous
    // coefficient the stop has at one `gap` of compression past its touch point, where its tangent
    // stiffness is `progression · rate`. Everywhere else the coefficient is this scaled by the local
    // tangent stiffness — `viscousCoefficient()` — so it is zero at the touch point and grows into
    // the stop with the material that is carrying load. A constant coefficient was the whole of
    // both measured defects: it put `damping · v` on the wheel in the tick contact became true
    // (0 → 28 kN at 0.7 m/s), and, stepped explicitly, it sat past the stability bound at rest. The
    // unit is unchanged and so is every authored value.
    double damping = 30000.0;

    // Rate-independent hysteresis, as a fraction of the stop's own elastic force opposing the
    // motion — the form a microcellular stop's dissipation is actually published in. A quasi-static
    // in-out stroke loses `2h/(1+h)` of the work it puts in, so BASF's 10-20 % band converts to
    // 0.053-0.111 here, and their Fig. 5 densities (11.9-13.1 %) to 0.063-0.070. Zero is the
    // shipped behaviour on every car, to the bit; unlike the viscous term it cannot spike on entry
    // (it scales with the elastic force, which starts at zero) and cannot trip the push-only clamp
    // on release (the factor stays above zero for any fraction below one).
    double hysteresis = 0.0;

    // How fast the stop must be moving for the hysteresis to be fully developed, metres per second.
    // The same numerical regularisation as `CornerSetup::damperFrictionSpeed` and for the same
    // reason: a hard sign term at 360 Hz makes a limit cycle rather than a dead band.
    //
    // Superseded where the Dahl branch below is stated, and that is the point of it: a real
    // bump stop's hysteresis develops over a **displacement**, which is a measured property, not
    // over a velocity, which is a number picked to keep a solver quiet.
    double hysteresisSpeed = 0.01;

    // --- the sourced dynamic branch, and every field of it is zero on every car in this project --
    //
    // Pech et al. measure a production passenger-car jounce bumper on a servo-hydraulic rig and fit
    // it as three parallel submodels: a static force-displacement characteristic, a **modified Dahl
    // friction element** whose saturation follows a displacement-dependent weighting function, and
    // **five Maxwell elements with their own contact points**. `rate`/`progression` above are the
    // first; these fields are the second and the third.
    //
    // Nothing here is stated on any car, so `statesDynamicBranch()` is false everywhere and the
    // arithmetic below is the arithmetic this model has always run. `jounceBumperCandidate` builds
    // a transferred parameter set from a stop's own static law; `stopdynamic 1` on a setup sheet
    // installs it for one session. **The transfer is from one published specimen and is graded as
    // such** — see that function and docs/suspension-fidelity-brief.md.

    // The Dahl coefficient, per metre, normalised so it is the reciprocal of the displacement the
    // friction force takes to build. `sigmaMin` at zero deflection rising to `sigmaMax` at
    // `dahlReference` through the source's own distortable exponent. **Zero disables the branch**,
    // which leaves `hysteresis` running through the `tanh` above exactly as it always did.
    //
    // The source publishes the *form* (its Equation 3) and does **not** publish its specimen's
    // three fitted values — Figure 12's right-hand panel has no numbers on its axis — so a
    // candidate set leaves min and max equal, which is the conventional constant-coefficient Dahl
    // and states nothing the source did not.
    double dahlSigmaMin = 0.0;
    double dahlSigmaMax = 0.0;
    double dahlSigmaExponent = 1.0;

    // The deflection the source's whole measurement programme is scaled by: on their specimen the
    // displacement at 9 kN, 71 mm. Metres. Only the Dahl's displacement dependence reads it.
    //
    // **The 71 mm is an inference and the paper never prints it** (corrected 2026-09-05). What the
    // paper states is a 68 mm figure and a 10 % displacement margin, which put the deflection at
    // 9 kN nearer 74.8 mm; 71 mm is the *hysteresis-monotonicity* limit, which its next sentence
    // places just below the turnaround. Carried because every fraction in `jounceBumperCandidate`
    // is a fraction of whatever this is, so changing it rescales the whole transferred element
    // table — a decision, not a typo fix.
    double dahlReference = 0.0;

    std::array<JounceElement, jounceElementCount> elements{};

    // Half-width of the C1 fade-in around each element's contact point, metres. A hard contact
    // point is a step in stiffness, and the source shapes it (their Figure 17, an order-3
    // polynomial through four constraints — which for these four is a quadratic). **Placed, not
    // sourced**: they optimise it and publish no value.
    double elementSmoothing = 0.0;

    // Whether this stop carries any of the branch above. A car that states none of it takes the
    // untouched `force()` path and never reads or writes a `TravelStopState`.
    [[nodiscard]] bool statesDynamicBranch() const
    {
        if (dahlSigmaMax > 0.0)
        {
            return true;
        }

        for (const auto& element : elements)
        {
            if (element.stiffness > 0.0 && element.cutoff > 0.0)
            {
                return true;
            }
        }

        return false;
    }

    // The static law alone, newtons: `rate · x^p / gap^(p−1)`, zero out of contact. **This is the
    // expression `force()` has always carried for its elastic term, character for character**, and
    // it is the whole of what the stop does at zero velocity — which is what holds the authored
    // curve fixed while the dynamic term around it changed (2026-09-08).
    [[nodiscard]] double elasticForce(const double past) const
    {
        if (past <= 0.0)
        {
            return 0.0;
        }

        return rate * std::pow(past, progression) / std::pow(std::max(gap, 1e-6), progression - 1.0);
    }

    // The viscous coefficient at this compression, N·s/m on the shaft: `damping` scaled by the
    // stop's own tangent stiffness, `k_t(x) / k_t(gap) = (x / gap)^(p−1)`.
    //
    // The statement is the textbook one for a lossy solid — the dissipative force is proportional
    // to the stiffness of the material that is deforming (Rayleigh's stiffness-proportional term;
    // for a Kelvin–Voigt solid `c / k` is the material's one relaxation time, here
    // `damping / (progression · rate)`, 14.8 ms on the Golf's bump stops). It says two things a
    // constant coefficient cannot: at the touch point nothing is deforming, so nothing dissipates
    // and the force *enters* from zero at any velocity; and deep in, where a strike lands, the
    // coefficient is the authored one and more. **No new number**: the scale is the stop's own
    // tangent stiffness and the length is the `gap` the elastic law is already written against.
    //
    // The push-only clamp is not here. At a fixed compression this term is linear in velocity, and
    // the corner solves it implicitly (`TravelStopTerms`); the clamp is applied there, to the
    // velocity the step arrives at, and in `force()` below for anyone holding the motion.
    [[nodiscard]] double viscousCoefficient(const double past) const
    {
        if (past <= 0.0 || damping <= 0.0)
        {
            return 0.0;
        }

        return damping * std::pow(past / std::max(gap, 1e-6), progression - 1.0);
    }

    // The two terms without the dynamic branch, which is what every car in this project runs.
    // `past` is how far into the stop the travel has gone, `rateOfChange` how fast it is going
    // further in; the hysteresis reads the latter the way it always did.
    [[nodiscard]] TravelStopTerms terms(const double past, const double rateOfChange) const
    {
        if (past <= 0.0)
        {
            return {};
        }

        const auto elastic = elasticForce(past);
        const auto hysteretic =
            hysteresis == 0.0 ? 0.0 : elastic * hysteresis * std::tanh(rateOfChange / hysteresisSpeed);

        return TravelStopTerms{.explicitForce = elastic + hysteretic, .viscousCoefficient = viscousCoefficient(past)};
    }

    // The same split with the dynamic branch running, advancing `state` by one tick. Only called
    // where `statesDynamicBranch()` is true, so `terms()` above stays the shipped path bit for bit.
    [[nodiscard]] TravelStopTerms dynamicTerms(double past, double rateOfChange, double deltaTime,
                                               TravelStopState& state) const;

    // The whole force at a stated velocity: the law as a function, for a test or a probe that holds
    // the motion and asks what the stop does. The corner does not call this — it solves the
    // viscous share against the velocity it is about to have rather than the one it had — but at
    // that velocity the two agree, and at zero velocity this is `elasticForce()` to the bit.
    //
    // The stop can only push: a viscous term large enough to go negative would have it pulling the
    // suspension back into itself as it releases.
    [[nodiscard]] double force(const double past, const double rateOfChange = 0.0) const
    {
        const auto split = terms(past, rateOfChange);

        return std::max(0.0, split.explicitForce + split.viscousCoefficient * rateOfChange);
    }

    // `force()` with the dynamic branch running. Same relationship to `dynamicTerms()`.
    [[nodiscard]] double dynamicForce(double past, double rateOfChange, double deltaTime, TravelStopState& state) const;
};

// The reference force the source scales its whole measurement programme by: 9 kN, chosen on their
// specimen as a 200% safety margin over the 3 kN a real road put through it. Their "reference
// displacement" is whatever deflection the bumper reaches this at, and every published fraction
// below is a fraction of that displacement.
export inline constexpr double jounceReferenceForce = 9000.0;

// A candidate dynamic branch for a stop, transferred from the source's specimen onto that stop's
// own static law.
//
// **The grade, stated first: this is one published specimen's fitted shape carried onto a stop
// whose static law is four and a half times shorter, and no jounce bumper of this car has been
// measured.** It is a candidate to measure with, not a car number. Nothing installs it by default.
//
// The transfer is the source's own scaling variable. Pech et al. set out to build "a generic
// procedure for jounce bumpers that differ in characteristic properties like the unstressed length
// or the stiffness", and the variable they scale everything by is the deflection at 9 kN. So this
// solves *this* stop's static law for its own deflection at 9 kN and places the elements at the
// same fractions of it, carrying the same force at the same fraction — which is the one transfer
// the source's own construction licenses. On the Golf's front bump stop that reference deflection
// is 15.9 mm against the specimen's 71.
//
// `cutoff` is left as a parameter because the source publishes **no fitted value** for it, only the
// band it must lie in: above the static ramps' excitation frequency and below the dynamic ones'.
// On their programme that is roughly 0.35 to 28 rad/s, and the default here is the geometric middle
// of it. It is a frequency, so it is carried across unscaled while the stiffnesses and contact
// points are not — a relaxation rate is a property of the material, not of how far it is squashed.
export [[nodiscard]] TravelStop jounceBumperCandidate(const TravelStop& stop, double cutoff = 3.1);

export struct CornerSetup
{
    CornerHardpoints hardpoints;

    // N/m along the damper axis, not at the wheel — the motion ratio is what turns one into the
    // other, and it varies with travel, so a wheel rate stated as a constant would be a lie the
    // geometry immediately contradicts.
    double springRate = 55000.0;
    double springFreeLength = 0.0;

    Curve damper = linearDamper(4200.0, 7600.0);

    // Seal and rod friction on the damper shaft, newtons, opposing the shaft's motion at any
    // non-zero velocity and not scaling with it. A `Curve` through the origin is a pure viscous
    // damper and has none; a real one carries a Coulomb term that dominates exactly where the
    // viscous term is smallest — small amplitudes and low velocities, which is straight-line running
    // on coarse tarmac, the first millimetre of a steering input, and the settling after a kerb.
    //
    // **Zero here, and zero on every car in this project, because nobody has sourced one.** Published
    // figures for a passenger car sit around 50 to 200 N at the shaft and a damper dynamometer plot
    // shows it directly as the width of the hysteresis loop at zero velocity — but AC's data has no
    // such number and none was found for this car (docs/suspension-fidelity-brief.md, item 5). The
    // mechanism is here so that the day a plot turns up it is a data change; putting an invented
    // number in would make it an architecture that asserts something nobody measured.
    double damperFriction = 0.0;

    // How fast the shaft has to be moving for the friction term to be fully developed, metres per
    // second. **A numerical choice and not a physical one**: the term is regularised as
    // `friction · tanh(velocity / this)` rather than `friction · sign(velocity)`, because a hard sign
    // term at 360 Hz makes a limit cycle rather than a dead band. Small enough to look like friction,
    // large enough that one tick cannot step across it.
    //
    // **It is not small against the measured friction's own structure, and that is worth knowing
    // before it is read as pure numerics** (2026-09-05). Deubel et al.'s steady-state curves put the
    // friction *maximum* of this class of strut at roughly 5 mm/s — half this width — so at the
    // velocities where a real damper's friction peaks, this `tanh` is still on its way up and is
    // delivering under half of `damperFriction`. The regularisation and the physics overlap.
    // `[.damper-friction-velocity]` measures it.
    double damperFrictionSpeed = 0.01;

    // **How that friction varies with shaft speed, normalised, against `|velocity|` in metres per
    // second.** Empty is the shipped law to the bit — the multiplier is exactly 1.0 at every
    // velocity and `solveDamperForce` runs the expression it always ran — and empty is what every
    // car in this project states.
    //
    // The magnitude above and this shape are kept apart deliberately, because their provenance is
    // not the same. `damperFriction` is one number per axle from one measurement (the front's is a
    // Passat B8 strut's quasi-static sliding friction, the rear's a compact-class monotube's
    // midstroke friction — two different papers and two different quantities). The shape is the
    // *steady-state velocity dependence* of the front strut alone, and there is **no rear
    // measurement of it at all**. Burying the two in one fitted curve would state a rear velocity
    // dependence that nobody has measured.
    //
    // The normalisation is stated rather than implied: the curve is 1.0 at
    // `damperFrictionReferenceSpeed`, which is the velocity `damperFriction` is quoted at. So
    // stating a shape does not restate the magnitude, and `macPhersonStrutFrictionShape()` is the
    // one sourced candidate. `Curve::at` holds its end values rather than extrapolating, which is
    // the wanted behaviour at both ends: below the source's slowest measured sliding velocity the
    // `tanh` above owns the answer, and above its 300 mm/s cap the model has no measurement and
    // must not invent a trend.
    Curve damperFrictionShape;

    TravelStop bumpStop;
    TravelStop droopStop;

    // The tire as a spring in series with the suspension, with the unsprung mass between them —
    // which is what makes wheel hop a mode the model has rather than one it cannot express.
    // 250 kN/m is a documented figure for a passenger radial at road pressure. Placeholder.
    double tireVerticalRate = 250000.0;
    double tireVerticalDamping = 1500.0;

    // --- the kerb-contact path's three numbers (docs/kerb-contact-brief.md) --------------------
    //
    // The tyre's section width, metres: the width of the cylinder the kerb-contact path collides
    // against the road, sidewall to sidewall. `ContactPatchSampling::width` is the *tread* band the
    // bottom grid samples and stays what it was; a kerb face meets the carcass, which is wider.
    // Placeholder: a 205 section, mid-size car.
    double tyreSectionWidth = 0.205;

    // The carcass's lateral rate, N/m — what a push on the **sidewall** meets, where
    // `tireVerticalRate` is what a push on the tread meets. The kerb-contact path blends the two
    // across the shoulder by the contact axis's angle to the wheel plane, and nothing else reads it.
    //
    // **Placeholder, and marked as one wherever it is stated.** No lateral measurement of a tyre
    // of this class exists in this project. The one published static measurement found
    // (Kulikowski & Szpica 2014, eleven used 12-14 inch tyres) puts the average lateral rate at
    // 65 N/mm against a radial 180 — a ratio of 0.36 — and that ratio applied to the vertical rate
    // above is what stands here until a measurement of the tyre does. Ledger in the brief.
    double tireLateralRate = 90000.0;

    // Coulomb friction of the sidewall against a kerb face, dimensionless. The tread's coefficient
    // for the same path is the tyre model's own longitudinal peak times the surface's grip; this is
    // the other compound. **Placeholder**: no measurement of sidewall rubber on concrete was found.
    // Tread rubber on dry concrete slides at 0.6-0.9 in the road-friction literature (HRB 1934)
    // and this sits inside that band.
    double sidewallFriction = 0.7;

    // **Lateral-force compliance steer**: how far this corner's wheel is twisted about the chassis's
    // up axis per newton of lateral force at its contact patch, radians per newton, **signed**.
    //
    // Every joint in `solveCorner` is an ideal pin or ball, so a rigid linkage takes no toe under
    // load at all. A real one carries rubber, and the toe it takes is not an imperfection — it is
    // *designed in*, because it puts the phase of the steering reaction force ahead of the steering
    // angle and that is most of what the rack feels like. Production cars are set up for a slight
    // **toe-out** under lateral force, which is a negative value here.
    //
    // The toe term and, below, the camber term are carried. A real bush also moves the upright
    // sideways and rearward, and those translations have no published figure behind them for this
    // class of car — so the hub stays where the linkage put it. `applyComplianceSteer` is the seam
    // and says why. docs/suspension-fidelity-brief.md, item 1.
    double lateralForceSteer = 0.0;

    // **Lateral-force compliance camber**: how far this corner's wheel is leaned about the
    // chassis's forward axis per newton of lateral force at its contact patch, radians per newton,
    // **signed**. Positive means the patch complies *with* the force — the contact point displaces
    // vehicle-inward under a cornering load and the tyre leans over it, which is what every
    // production car measured does and is the adverse direction: the loaded outside wheel leans out
    // of the turn.
    //
    // What it reaches in this model is the *geometry* and not the force law: the tilted spin axis
    // moves the sampled contact patch and redistributes penetration across its width, and the
    // constructed patch point moves the lever arm the geometric load path carries the tyre force
    // through. `evaluateTyre` has no camber input, so there is no camber thrust and no
    // camber-dependent grip — stated here so nobody reads this coefficient as one.
    // `applyComplianceCamber` is the seam. docs/suspension-fidelity-brief.md, item 1.
    double lateralForceCamber = 0.0;

    // **Longitudinal recession**: how far this corner's wheel centre displaces along the chassis's
    // forward axis per newton of longitudinal force at its contact patch, metres per newton,
    // **signed**. Positive means the wheel complies *with* the force — rearward under braking,
    // forward under traction — which is what every production axle is built for: the fore-aft
    // bushing is the soft one, because impact harshness rides it.
    //
    // **No measurement of any car exists for this channel.** What exists is a design-target band —
    // Heissing & Ersoy, *Chassis Handbook* (2011), Table 1-6: front **4–8 mm/kN of braking
    // force**; rear **8–16 mm PER G of deceleration** (the table's own footnote — a different
    // unit, folding in the car's rear brake share) — and a design target is what a manufacturer
    // aims at, not what a rig read. The default is 0.0; the Golf states the band's middle since
    // 2026-08-29 night, **on Dominic's word after a driven A/B**, with the grade flagged where it
    // is stated. `front.recession` / `rear.recession` on the setup sheet (mm/kN) are the A/B.
    //
    // What it reaches is the wheel's *position* and not the force law: the sampled patch grid and
    // the applied-force point move with the hub, so the vertical load's pitch lever and the road
    // the patch reads both shift. **And through the patch it reaches the steering weight** — a
    // claim this comment originally got wrong, corrected 2026-08-29 night after the seat found
    // it: the rack torque is the tyre resultant's moment about the kingpin axis applied at the
    // patch the solve reports, and the kingpin does not recede with the hub, so braking grows the
    // mechanical trail by the recession (~30 mm against this car's 36) and power shrinks it.
    // Measured on the seat A/B: rack force per g of lateral +28% under trail braking, −32% on
    // power. What still sees nothing: the linkage Jacobians and the tyre's force law.
    // `applyComplianceRecession` is the seam and says why. The force-to-displacement map
    // saturates at ±50 mm (`recessionDisplacement`), because the linear coefficient is only
    // claimed inside the band's own context and a kerb strike's one-tick 20 kN spike must not
    // teleport the hub a fifth of a metre.
    double longitudinalForceRecession = 0.0;

    // Placeholder: a hub, upright, brake and wheel for a mid-size car.
    double unsprungMass = 38.0;

    // Torsional rate of this corner's half of the anti-roll bar, N·m per radian of *difference*
    // across the axle, expressed at the wheel as N/m of differential travel. Zero disables it.
    //
    // **The number keeps that meaning whether or not the corner states a drop link**, which is the
    // whole of how the geometry was added without moving every car's roll stiffness. A corner with
    // drop-link hardpoints refers this wheel rate onto the link through the link's own motion ratio
    // *at design* — `k_link = k_wheel / ratio²`, the standard referral — so at the design position
    // the bar is worth exactly what it was worth before, and what the geometry buys is that the
    // ratio then varies across the travel and the force rides the link's own Jacobian instead of the
    // wheel's. See `solveAntiRollBar`.
    double antiRollRate = 0.0;

    // Rolling resistance, as a fraction of the vertical load. Applied as a torque on the wheel
    // rather than as a force at the patch, because that is what it physically is: the contact
    // pressure is higher at the leading edge of the patch than the trailing one, and the resultant
    // acts ahead of the wheel centre. Placeholder: 0.012 for a road radial.
    double rollingResistance = 0.012;

    TyreModel tyre;

    // Placeholder: wheel, tire, hub and disc for a mid-size car.
    double wheelInertia = 1.2;

    // What this corner's brake makes at a **fully applied pedal**, N·m — after the hydraulics and
    // after whatever the rear circuit's valve does to them, so it is the number a corner is actually
    // worth rather than one an axle's share has to be applied to.
    //
    // Placeholder here. A car with real hardware derives it (`peakBrakeTorque` in `:Brakes`), which
    // is what took `brakes.ini`'s `MAX_TORQUE` and `FRONT_SHARE` out of the Golf on 2026-08-23; the
    // setup sheet can still override it, because a pad change is a setup change.
    double brakeTorque = 1400.0;

    // The disc as a lump of iron and the pad's fade curve with it — `brakeThermalOf` derives the
    // whole thing from the same `BrakeHardware` that derived the torque above, so nothing here is a
    // second statement of a part. Inert unless `VehicleSetup::brakeThermal` switches it on, and inert
    // whatever that says on a brake whose couple states no fade curve.
    BrakeThermal disc{};

    // The wheel the disc is bolted inside — the third node, and the only path this model has from a
    // brake at 500 °C to a tread at 50. `wheelThermalOf` derives it from a `WheelHardware` and the
    // same `BrakeHardware` the disc came from, so the hat that joins them is stated once.
    //
    // **A default-constructed one has no heat capacity, which means no wheel.** Every step treats
    // that as absent, so a car that does not state a wheel heats exactly as it did before stage 3
    // existed. docs/brake-thermal-brief.md, section 7.
    WheelThermal wheel{};
};

// One aerodynamic surface: a body, a wing, a splitter. Its coefficients are constants this
// milestone — the deferred model replaces each with a lookup against ride height and rake, and
// nothing else moves, because the ride height it would be looked up against is already an input to
// where this is evaluated and the point it acts at is already data.
export struct AeroSurface
{
    // Where it acts, in the chassis frame. Data rather than the centre of gravity, because a
    // splitter and a rear wing pitch the car in opposite directions and that is most of the point.
    glm::dvec3 centre{0.0};

    // Drag area, Cd*A in square metres. Placeholder: 0.30 by 2.2 m2 for a mid-size sedan.
    double dragArea = 0.66;
    // Lift area, Cl*A. Positive lifts, negative presses down. A road sedan makes a little lift.
    double liftArea = 0.10;
};

export struct VehicleSetup
{
    std::array<CornerSetup, cornerCount> corners{};

    // The sprung mass ledger. Chassis, fuel and occupants as separate entries because that is the
    // seam a fuel model needs — mass, centre of gravity and inertia are derived from this every
    // time it changes rather than authored once.
    std::vector<MassComponent> sprung;

    ContactPatchSampling sampling{};

    // The brake system between the pedal and the four corners above.
    //
    // **This is where the other half of the brake bias lives**, and the half a set of calipers cannot
    // express: `CornerSetup::brakeTorque` states what each corner makes at full pedal, and these two
    // state how the pedal gets there — which is not the same shape on both axles once a valve is
    // fitted. Defaults are a car with no servo and no valve, so the pedal maps linearly onto each
    // corner's peak, which is exactly what this model did before either existed.
    BrakeHydraulics brakeHydraulics;
    ProportioningValve rearBrakeValve;

    std::vector<AeroSurface> aero;

    // The body's collision shape and what it does when it hits something. Placeholder: a box a
    // little smaller than a hatchback, so the wheels reach the ground before the bodywork does.
    CollisionBox body;
    ContactMaterial contact;
    // Sea level, 15 degrees. Placeholder, and the one number here a weather model would move.
    double airDensity = 1.225;

    // Steering rack travel per unit of steering input, metres, **signed**: positive demand is a
    // right turn, and whether that needs the rack going positive or negative is a property of the
    // linkage — a steering arm ahead of the kingpin steers the opposite way to one behind it. Every
    // car has to state or derive it; this default is a magnitude with a sign that means nothing, and
    // a setup that leaves it alone has not answered the question.
    double rackTravelPerInput = 0.055;

    // Where this car's ride height is quoted from: the height above the *design contact plane* of
    // the point on the chassis a ride height figure refers to, metres. The body frame puts the
    // design contact plane at y = 0, so this is a ride height at design attitude by construction.
    //
    // **Stated rather than inferred, and the first attempt at inferring it is why.** Taking the
    // underside of the collision box looks principled and is wrong on the one car with real data:
    // AC's `COLLIDER_0` is a coarse body shell 0.32 m up, not a floor, so the Golf reported a 0.30 m
    // ride height — a correct measurement of the wrong surface. A car's ride height is a figure the
    // car states, and a model that derives it from a collision shape derives it from whatever that
    // shape happened to be authored as.
    //
    // Placeholder, like every vehicle figure here, and a candidate for exactly the parameter sweep
    // this file's units are getting.
    double rideHeightReference = 0.12;

    // What the steering *wheel* turns lock to lock, radians. 13.195 rad is 756 degrees, which is
    // 2.1 turns and is this class of car's rack.
    //
    // The vehicle model needs it for exactly one thing and it is not a force: **the telemetry's
    // `Steering Angle` column**, which until 2026-08-21 carried `input.steering` — a dimensionless
    // demand from −1 to 1 — multiplied by 57.2958 and labelled degrees. That is the `Engine RPM`
    // failure exactly: a units conversion applied to a quantity with no units. A demand is not an
    // angle, and turning one into the other needs a number only the steering box has.
    double steeringLockToLock = 13.194689145077131;

    // Whether the tyre's in-plane forces reach the corner's degree of freedom through the linkage's
    // own Jacobian — the geometric load path: roll centre, jacking, anti-dive, anti-squat and
    // anti-lift, which are not four features but four readings of one dot product.
    //
    // Off is the model this project measured everything against: the tyre force reaches the corner
    // as its vertical component times the wheel's vertical Jacobian, so **every newton of load
    // transfer deflects a spring** and none of it travels through the wishbones. That is why the car
    // leans about 12% more than its own springs and bars would make it lean, and why it has exactly
    // zero anti-dive. Total load transfer and longitudinal balance are unaffected either way — those
    // are fixed by the whole-car free body whatever path the force takes inside a corner — so this
    // is an **attitude** switch with a balance side effect, which is the shape the seat reported.
    //
    // A switch rather than a rewrite because the old behaviour is what every measured figure in
    // docs/ was taken under, and because it is the control: `docs/suspension-load-path-brief.md`
    // stage 1 requires both parity gates byte-identical with this off.
    bool geometricLoadPath = false;

    // Whether the wheels' spin is allowed to react on the rest of the car — the driveline's own
    // Newton's third law, in both the places it acts.
    //
    // On the **chassis**: a wheel being spun up or slowed down takes angular momentum from
    // somewhere, and the model gives it none. The chassis receives the tyre force at the contact
    // patch, which is exactly right whenever the wheel's spin is steady, and is short by the wheel's
    // own `I·alpha` whenever it is not — a launch, a shift, a lock-up. Switched on, each corner puts
    // `−I·alpha` back on the body about the wheel's spin axis, which is the term that makes the
    // whole car's angular momentum balance close (docs/suspension-fidelity-brief.md, item 3A).
    //
    // In the **corner**: a chassis-mounted transaxle drives the hub, so the shaft torque does virtual
    // work in the corner's own coordinate as the upright turns about the wheel's spin axis. That term
    // is what converts an outboard brake's contact-patch force line into an inboard drive's
    // wheel-centre line — the textbook distinction, and anti-lift under power on a driven axle. An
    // outboard brake needs no counterpart: its couple is internal to the wheel assembly.
    //
    // Off by default, and off is the model every figure in docs/ was measured under.
    bool drivelineReaction = false;

    // Whether the tyres carry a temperature — the tread's own heat balance, and grip following the
    // tread **core** through the compound's own curve (`TyreThermal`).
    //
    // Off is the tyre this project measured everything against: friction, vertical rate and rolling
    // resistance all constant from the moment the car is built to the end of the session, and a tyre
    // that is therefore always exactly at its best. **Every performance figure in docs/ was taken
    // under that assumption**, and the moment temperature is a state every fixture acquires a
    // precondition it does not state — which is why the default seeds at
    // `tyreDefaultTemperature`, the middle of the curve's flat plateau, so that switching this on
    // changes nothing until something heats or cools a tyre.
    //
    // A switch rather than a rewrite for `geometricLoadPath`'s reason exactly: off is the control,
    // and `OSR_TYRE_THERMAL=off` is the way back from the seat without a rebuild.
    // docs/tyre-state-brief.md.
    bool tyreThermal = false;

    // Whether the air inside the tyres carries a temperature and a pressure, and the carcass's
    // vertical rate and rolling resistance follow it.
    //
    // **Second to `tyreThermal` and dependent on it in practice**, because pressure is the gas law
    // applied to a temperature and there is nothing to apply it to until the tread has one. On with
    // the tread's model off, the gas simply follows a carcass that never changes and the pressure
    // never moves off its seed — which is a defensible state to be in and is not a useful one.
    //
    // Off is every performance figure this project has: a tyre permanently at its ideal pressure,
    // which is what `CornerSetup::tireVerticalRate` and `CornerSetup::rollingResistance` are quoted
    // at. `OSR_TYRE_PRESSURE=on|off` is the seat control. docs/tyre-state-brief.md, section 7.
    bool tyrePressure = false;

    // Whether the brake discs carry a temperature, and the pads' friction follows it — which is fade.
    //
    // Off is the brake every figure in docs/ was measured on: a coefficient that is the same on the
    // first stop of the day and the tenth in a row. **The inertness proof comes from the opposite end
    // of the curve to the tyre's** and that is worth noticing rather than copying blindly — a pad's
    // rated friction is flat *cold*, so seeding a disc at ambient is both the physical seed and the
    // one that changes nothing, where the tyre had to be seeded warm to sit on its plateau.
    //
    // `OSR_BRAKE_THERMAL=off` is the control and the way back. docs/brake-thermal-brief.md.
    bool brakeThermal = false;

    // Whether each tyre is also collided against the road as a cylinder, so that a kerb **face** can
    // push on it — the kerb-contact path. The bottom grid samples the road with vertical rays and
    // cannot see a face steeper than its own pitch: GCP's 150 mm kerbs arrive as a one-tick road
    // step, 30 kN on the tyre and the wheel into its stop at 4 m/s (docs/kerb-contact-brief.md).
    //
    // On, every contact the cylinder reports within 15 degrees of the wheel plane's down direction
    // is dropped — those are the bottom grid's, on flat road, on banking, on the Bathurst chamfer —
    // and each survivor is a spring contact on the carcass with Coulomb friction, reaching the
    // corner's degree of freedom, the chassis at the contact point and the wheel's spin. Nothing
    // enters the belt bed, the patch normal, the patch penetration or the tyre model.
    //
    // Off is the car every figure in docs/ was measured on and the control; **on must keep both
    // parity goldens byte-identical**, through the exclusion rule, and that is the first thing the
    // switch has to prove. `OSR_KERB_CONTACT=on|off` is the seat control.
    bool kerbContact = false;

    // Whether each corner's equation carries the chassis's own acceleration at the wheel's
    // attachment — the unsprung mass's inertial term in the moving body frame (2026-09-08 latest of
    // all, docs/frame-acceleration-brief.md).
    //
    // The corner's coordinate is relative to the chassis: the wheel centre is `x_B + R·(C(q) − c)`,
    // so the unsprung mass's absolute acceleration is the attachment point's plus `R·C'·q̈` (plus
    // `R·C''·q̇²`, and a Coriolis term the projection onto `R·C'` annihilates). Projected onto the
    // corner's own Jacobian, Newton's law for the unsprung mass reads
    //
    //     m_u·|C'|²·q̈ = Q_elements + Q_tyre − m_u·(R·C')·(a_attach − g)
    //
    // where `a_attach = a_B + α × r + ω × (ω × r)` is the acceleration of the body-fixed point under
    // the wheel centre. The model carried `−m_u·g·C'_y` — the weight — and nothing for `a_attach`,
    // which is the same as saying the chassis is infinitely massive: in the air every hanging wheel
    // pulled the body down with its full weight (629 N at the Golf's rear droop limit, where the
    // spring less the stop is 207), and a corner under a body accelerating at 1 g saw its wheel at
    // the wrong weight by exactly that. On, the term is solved **coupled to the chassis's own
    // response to the four reactions** — a Gauss–Seidel sweep over the corners with the rigid body's
    // mobility at each wheel centre, `attachmentMobility` — which is what makes the corner–chassis
    // pair one two-body system: a corner's inertia against a free body is the reduced one,
    // `m_u·|C'|²·(1 − m_u·G)`, and a car in free fall keeps its wheels exactly where they are.
    //
    // The chassis's side — the reaction `−m_u·q̈·R·C'` at the wheel centre — is untouched: that
    // equation was already the exact one, which is why the momentum ledger closed before this. The
    // weight's flat-world projection, `−m_u·g·C'_y` in the body frame, is replaced on the geometric
    // path by the exact one on the same Jacobian, `m_u·g·(R·C')`, because the invariance the term
    // exists for — a uniformly accelerating car has no relative suspension motion — holds only with
    // both halves on one Jacobian; the two are the same bits on a level body. Off is the model every
    // figure in docs/ older than 2026-09-08 latest of all was measured on, and the control.
    // `OSR_FRAME_ACCELERATION=on|off` is the seat control.
    bool frameAcceleration = true;

    // Whether each corner's equation carries its own configuration-dependent inertia term
    // (2026-09-08 latest of all (c), docs/nonlinear-geometry-brief.md). The corner's generalised
    // inertia is `I(q) = m_u·|C'(q)|²` and it varies with travel, so Lagrange's equation from
    // `T = ½·I(q)·q̇²` reads `I·q̈ + ½·I'(q)·q̇² = Q` — the same `−m_u·(C'·C'')·q̇²` the unsprung mass's
    // acceleration carries, since `I' = 2·m_u·C'·C''`. Without it a mass sliding along a curve whose
    // parametrisation stretches speeds up or slows down in the world for no force. On, the term is
    // `−½·I'(q₀)·q̇₀²` in the explicit generalised force, `I'` differenced over ±`inertiaSlopeStep`
    // from the same solve the inertia is read from (`cornerInertiaSlope`). It is independent of the
    // chassis's motion — the projection onto `R·C'` is rotation-invariant — so it neither touches nor
    // duplicates the frame terms. Exactly `−0.0` wherever `|C'|` does not vary with travel. Off is the
    // corner before this change and the control; test-facing, no sandbox knob.
    bool nonlinearGeometry = true;

    // Whether the chassis receives the whole of each unsprung mass's acceleration relative to its
    // design-point rigid motion, and not only the `−m_u·q̈·R·C'` of the corner's own coordinate
    // (2026-09-08 latest of all (d), docs/chassis-kinematic-reaction-brief.md). The chassis body
    // carries every unsprung mass rigidly at its **design** wheel centre; the true mass is at
    // `x_B + R·(C(q) − c)` with velocity `v_B + ω × r + R·C'·q̇`, and the difference in momentum,
    // `Σ m_u·(ẋ_u − ẋ_u^design)`, has a rate the chassis must be handed for the whole car's momentum
    // and angular momentum to be what the true system's are:
    //
    //     F = −m_u·[ α × Δr + ω × (ω × Δr) + 2·ω × R·C'·q̇ + R·C''·q̇² ]    at the wheel centre
    //     τ = −m_u·Δr × (a_design − g)                                       a pure couple
    //
    // with `Δr = r − r_design` the wheel's offset from its design point and `a_design` the design
    // point's own rigid acceleration. Before this the chassis received `−m_u·q̈·R·C'` alone: the
    // centripetal pull of a wheel on its arc (`R·C''·q̇²`, hundreds of newtons at the 60 mm dip's
    // rates), the Coriolis force of a wheel moving on a turning body, and the inertial force of a
    // displaced wheel were all left out of the body's balance, and a car alone in the air did not
    // keep its momentum. `unsprungKinematicReaction` is the statement; it is solved inside the same
    // sweep as the frame term (its offset terms depend on the chassis's own acceleration), and it
    // is exactly zero at rest on a level road. Off is the chassis before this change and the
    // control; test-facing, no sandbox knob.
    bool kinematicReaction = false;

    [[nodiscard]] double unsprungMass() const
    {
        auto total = 0.0;
        for (const auto& corner : corners)
        {
            total += corner.unsprungMass;
        }

        return total;
    }
};

// The wheel inertias in corner order, from the one place they are stated. The driveline needs them
// to refer an axle's inertia through the gearing, and a caller assembling that array by hand is a
// second statement of a number that already has an owner — which is what it was, and nothing made
// the two agree.
export [[nodiscard]] std::array<double, cornerCount> wheelInertias(const VehicleSetup& setup);

// What the caliper at each corner is at, pascals, for a given brake pedal. The master cylinder's own
// characteristic for the front pair and the proportioning valve's output for the rear.
//
// It is a function of the *car* rather than of a corner because the two axles share one master
// cylinder, which is the whole reason a fixed bias cannot be right everywhere.
export [[nodiscard]] std::array<double, cornerCount> brakeCircuitPressures(const VehicleSetup& setup,
                                                                           const double pedal);

// The fraction of `CornerSetup::brakeTorque` a corner makes at a given pedal — that corner's pressure
// against its own pressure at a fully applied pedal.
//
// **This is what makes the bias pedal-dependent instead of a constant ratio.** Both axles are linear
// in it below the valve's knee and the servo's runout; above either, they are not, and they are not
// in the same way. With no servo and no valve it is the pedal itself, to the bit.
export [[nodiscard]] double brakePedalResponse(const VehicleSetup& setup, const std::size_t corner, const double pedal);

// The fore-aft bush's force-to-displacement map: linear at the stated coefficient, saturated at
// ±50 mm. The guard is the stage-2b floor's argument over again — a published form used outside
// its data needs a stated bound, not trust. Heissing/Ersoy's band is quoted for braking forces (a
// full-pedal front wheel carries about 5.5 kN, and 8 mm/kN of that is 44 mm), while a kerb strike
// puts a one-tick ±20 kN spike through the patch, four times outside the band's context; measured
// unguarded on the scripted launch, that read ±222 mm of hub displacement — no bushing's travel.
// Fifty millimetres is the band's own in-context edge rounded up, a guard against extrapolation
// and not a modelled bump stop. Inside it the map is the coefficient times the force, exactly.
//
// In the interface because it is constexpr; the state write in `VehicleImpl.cpp` is the one
// consumer, so the trace's `Recession` column and the applied geometry cannot disagree.
export [[nodiscard]] constexpr double recessionDisplacement(const double recessionPerNewton, const double force)
{
    constexpr auto travelLimit = 0.05;
    const auto displacement = recessionPerNewton * force;

    return displacement > travelLimit ? travelLimit : displacement < -travelLimit ? -travelLimit : displacement;
}

export struct CornerState
{
    // The corner's whole degree of freedom: where the lower wishbone is, and how fast it is moving.
    double wishboneAngle = 0.0;
    double wishboneRate = 0.0;
    // Wheel spin, integrated in the same loop as everything else. Nothing drives it yet — the road
    // does, through the tire, which is enough to roll and to brake.
    double wheelSpeed = 0.0;

    // The carcass deflections, which are what make the tire transient rather than instantaneous.
    TyreState tyre;

    // How far the bushes have twisted this corner about the chassis's up axis, radians. State rather
    // than a derived quantity, because it is driven by the tyre force and the tyre force is driven by
    // the toe: solving that inside one tick means an iteration, and this model's only iteration is
    // the contact patch's.
    //
    // Carried from the previous tick instead, which is one tick of explicit lag at 360 Hz. **That was
    // refused for the jacking force and is accepted here, and the difference is the sign of the
    // loop**: more toe-out lowers the slip angle, which lowers the force, which lowers the toe-out,
    // so the lag sits inside negative feedback and settles. Jacking is the other way round. A rubber
    // bush also has a real relaxation of about this order, so what the lag models is not nothing —
    // but nobody has sourced its time constant, so it is not claimed as one.
    double complianceSteer = 0.0;

    // And how far they have leaned it about the chassis's forward axis, radians — the camber twin
    // of the field above, carried from the previous tick for the same reason. The lag is even
    // safer here: camber never reaches the tyre's force law (see `lateralForceCamber`), so the loop
    // this sits in is not merely negative feedback, it is very nearly open.
    double complianceCamber = 0.0;

    // And how far they have let it recede along the chassis's forward axis, metres — the
    // translation third of the same bush, carried from the previous tick for the same reason. The
    // lag argument is the camber's: recession never reaches the tyre's force law, so the loop is
    // very nearly open.
    double complianceRecession = 0.0;

    // The brake disc's temperature, degrees Celsius. Seeded at `brakeDefaultTemperature`, which is
    // cold — see `VehicleSetup::brakeThermal` for why cold is the inert seed here and warm was the
    // inert seed for the tyre.
    double discTemperature = brakeDefaultTemperature;

    // The wheel's, the same way. **A disc temperature without its wheel is not a state**, which is
    // why `seedDiscTemperatures` sets both: a fixture measuring a first stop of the day is asserting
    // that the whole corner is cold, and a wheel left at whatever the last run finished on would
    // carry a stint's worth of heat into it.
    double wheelTemperature = brakeDefaultTemperature;

    // What the **bump** stop's dynamic branch remembers. Written only by a corner whose stop
    // states that branch, which is no car in this project, so on every shipped car these bytes are
    // the zeros they were constructed with and stay them for the whole session.
    //
    // The droop stop deliberately has no state and no branch: on a strut it is the damper topping
    // out, a different mechanism with its own account, and the source measured a jounce bumper.
    TravelStopState bumpStopState;
};

// The vehicle's own state and nothing else's. Engine speed lived here until the driveline was given
// a state of its own: it was written by `stepDriveline`, read by nobody in this file but the
// telemetry fill, and its being here made `:Vehicle` the keeper of a number belonging to a partition
// it does not import. `DrivelineState` in `:Driveline` owns it now, and the caller that steps both
// is what joins them.
export struct VehicleState
{
    RigidBodyState chassis;
    std::array<CornerState, cornerCount> corners{};
};

// Put every one of a car's brake discs at a stated temperature. A fixture measuring a tenth stop in a
// row starts where the ninth left off; one measuring a first stop of the day says so with this.
export void seedDiscTemperatures(VehicleState& state, const double celsius);

// Put every one of a car's tyres at a stated temperature. What a fixture calls when it wants a cold
// car — and the reason it is spelled at the call rather than defaulted is that a starting temperature
// is part of a measurement: a stop from 20 °C and a stop from 85 °C are different experiments, and a
// fixture that does not say which one it is running has not stated its own preconditions.
export void seedTyreTemperatures(VehicleState& state, const double celsius);

// Put every tyre's cavity air at the temperature that makes it read its own ideal pressure.
//
// **What a fixture calls when it does not want to be measuring a pressure change.** Every
// pressure-dependent number a car states is quoted at the ideal, so a car seeded here behaves exactly
// as it did before `tyrePressure` existed — and that is the inertness proof, asserted rather than
// hoped for. It takes the setup because the temperature is a property of the car's own cold and ideal
// pressures and not of its state.
export void seedTyreGasPressures(const VehicleSetup& setup, VehicleState& state);

// `raceengine.assists` restates its own wheel count rather than importing this one, because it must
// not be able to import anything from this module — see the head of `Assists/Api/WheelSensors.cppm`.
// This is the join: two statements of one number that the compiler will not let disagree.
static_assert(wheelCount == cornerCount, "the assist layer and the vehicle must agree on how many wheels a car has");

static_assert(std::is_trivially_copyable_v<VehicleState>, "the harness saves and restores this by copying its bytes");
static_assert(std::is_standard_layout_v<VehicleState>, "and rollback will later");

export struct VehicleInput
{
    // -1 to 1, and 0 to 1. Sampled once per physics tick and never polled from inside the model.
    double steering = 0.0;
    double throttle = 0.0;
    double brake = 0.0;
    // 0 with a foot off it and 1 fully depressed, which is the way round the pedal works: pressing
    // it *releases* the clutch. The auto-clutch layer fills it in when nobody is on it, and a driver
    // past the pedal's free play takes it back outright.
    double clutch = 0.0;
    std::int32_t gear = 0;
};

// Nothing driving the wheels: a car being pushed, coasted or dropped, and every tick taken before a
// driveline exists. Named rather than written `{}` at the call, because a bare brace there reads as
// an omission and this is a statement.
export inline constexpr std::array<double, cornerCount> noDriveTorque{};

// How many distinct obstacle faces one wheel may carry in a tick after the exclusion rule and the
// per-face merge. A kerb is one face and its top edge; a wheel wedged in a gutter is two. Eight is
// a ceiling nothing on either track reaches, and the array is fixed so that the tick allocates
// nothing for it.
export inline constexpr std::size_t maxObstacleContacts = 8;

// The exclusion cone: a contact whose axis is within this angle of **the grid's own normal** — the
// contact patch's `normal`, the pressure-weighted mean of what the vertical rays hit, or the wheel
// plane's up (the negation of `contactPatchSamples`'s `inPlaneDown`) when the wheel is in the air —
// is the bottom grid's contact and is dropped. So is any contact against a triangle whose own face
// normal is within it, whatever the push-out points along, because that triangle is road the grid
// carries and its edges are triangulation rather than surface.
//
// Fifteen degrees, cos 15. **Bookkeeping between two models of one tyre, not physics**: the angle
// is chosen so that every surface the goldens drive over — flat road, banking, the 9.46 degree
// Bathurst chamfer, the E2 crest — is on the grid's side of it, and a surface steeper than it has
// no grid contact to double-count against.
//
// **Why the grid's normal and not the wheel's up** (2026-09-08, found by the full suite): on a
// longitudinal grade the wheel plane contains the world's down, so the wheel's "up" is the world's
// and a 15 degree hill sits exactly on the cone — the two slope fixtures went red on the boundary's
// rounding. The grid carries a hill the way it carries banking, and its own normal says so.
export inline constexpr double obstacleExclusionCosine = 0.96592582628906829;

// Where the fade towards that cone begins: a contact whose axis is further than thirty degrees
// from the wheel's up carries full weight, and between thirty and fifteen its force is scaled down
// linearly in the cosine, to zero at the cone. Bookkeeping at the seam between the two models, and
// measured before it existed: a wheel mounting a 150 mm step at 33 kph had its face contact cross
// the cone while bouncing on its bump stop, and a 40 kN force switching on and off across three
// ticks is a step the grid on the far side of the seam did not make up for. Cos 30.
export inline constexpr double obstacleFullWeightCosine = 0.86602540378443865;

// Two survivors whose axes lie within this angle of each other are the same face reported by two
// triangles, and only the deeper is kept. Five degrees, cos 5 — the same threshold the mesh uses to
// decide that two neighbouring triangles are one surface.
export inline constexpr double obstacleMergeCosine = 0.99619469809174553;

// The rounding on the cylinder's two rims, metres — Jolt's default convex radius, restated here
// because the rule below reconstructs the cylinder's support point and has to agree with the shape
// the backend built. The backend lowers it where a stated extent is smaller, and so does the rule.
export inline constexpr double obstacleConvexRadius = 0.05;

// How far a contact's reported depth may disagree with the cylinder's own protrusion past the
// contact plane before the contact is rejected as not being a push-out at all: two millimetres
// plus two per cent. A true minimum-translation contact agrees to EPA's tolerance; the case this
// exists for disagrees by centimetres — see `keepObstacleContacts`.
export inline constexpr double obstacleDepthTolerance = 0.002;

// The exclusion rule, the depth-consistency check, the hub-height clause and the per-face merge,
// pure and total: `found` is what `collideCylinders` reported for one wheel, `wheel` is the
// cylinder it was reported against, and what comes back in `kept` is the set of distinct faces the
// bottom grid is not already carrying, with how many were written. Exported so the rule can be
// tested against a generated mesh without standing up a vehicle.
//
// The hub-height clause: a push whose contact point is at or above the wheel centre, in the wheel's
// own up direction, is a wall and not a kerb — a tyre cannot climb what its hub is not above — and
// a wall is the chassis collider's to resolve. Dropped here so the bodywork's contact and the tyre's
// spring do not both answer the same wall.
//
// `gridNormal` is the contact patch's normal when the wheel is touching and zero when it is not;
// zero falls back to the wheel plane's up. The hub-height clause reads the wheel's up regardless.
export [[nodiscard]] std::uint32_t keepObstacleContacts(std::span<const ObstacleContact> found,
                                                        const WheelCylinder& wheel, const glm::dvec3& gridNormal,
                                                        std::array<ObstacleContact, maxObstacleContacts>& kept);

// Which of a corner's two authored range limits held it on a tick, if either did
// (2026-09-08 later still, docs/range-constraint-brief.md; `constrainCornerRange` below).
export enum class RangeLimit
{
    None,
    Bump,
    Droop
};

export struct CornerForces
{
    double spring = 0.0;
    double damper = 0.0;
    double bumpStop = 0.0;
    double droopStop = 0.0;

    // The kerb-contact path's share of the load that reached this corner's degree of freedom, as
    // the equivalent force at the wheel, newtons. **Not part of `tireVertical`**, which is also the
    // tyre model's load, the rolling-resistance load, the tread's thermal load and the `Tyre Load`
    // channel, none of which a kerb face is allowed to enter. Exactly 0.0 with `kerbContact` off.
    double obstacleTravel = 0.0;

    // **Always the equivalent force at the wheel**, newtons, whichever bar model this corner is on —
    // negative on the wheel that is further into bump, because the bar pushes it back out. A corner
    // that states a drop link computes its force along the link and reports the wheel force that
    // would do the same virtual work, so this field means one thing and telemetry reading it does
    // not have to know which model a car is on.
    double antiRoll = 0.0;

    // And the force along the drop link itself, newtons, zero on a corner that states none. The two
    // corners of an axle carry this exactly equal and opposite — a torsion bar is one internal force
    // — which is **not** true of `antiRoll` above once the link has a motion ratio, because the two
    // corners sit at different points in their travel and so convert it differently.
    double antiRollLink = 0.0;

    double tireVertical = 0.0;

    // The linkage's range constraint, as the equivalent force at the wheel, newtons, positive toward
    // bump — so it is negative on a corner held at its bump limit and positive on one hanging from
    // its droop limit (2026-09-08 later still, docs/range-constraint-brief.md). The residual the
    // authored range carries once the spring, the stops' elastic laws, the damper at the held
    // velocity and the road have been counted; never a force sized from a penetration or a
    // stiffness. **Exactly 0.0 on every tick the corner's coordinate stays inside its range.**
    double rangeLimit = 0.0;

    // The chassis's own acceleration at this wheel's attachment, as the equivalent force at the
    // wheel, newtons, positive toward bump (2026-09-08 latest of all,
    // docs/frame-acceleration-brief.md): `−m_u·(R·C')·a_attach` over `travelPerAngle`, with the
    // weight's exact projection folded in on the geometric path, at the rates that acted — the four
    // corners' reactions of this tick included. A body falling at g reads about `+m_u·g` here, the
    // wheel's weight given back; a body at rest on the road reads 0.0 to rounding. **Exactly 0.0
    // with `VehicleSetup::frameAcceleration` off.**
    double frame = 0.0;

    // The corner's configuration-dependent inertia term, `−½·I'(q)·q̇²`, as the equivalent force at the
    // wheel, newtons, positive toward bump (2026-09-08 latest of all (c),
    // docs/nonlinear-geometry-brief.md). Even in the rate: the same sign whichever way the wheel moves.
    // On the Golf a few newtons at the characterisation's largest rates; exactly ±0.0 at rest, wherever
    // `|C'|` does not vary with travel, and with `VehicleSetup::nonlinearGeometry` off.
    double geometry = 0.0;
};

// The tire's frame at the contact patch, and what it is doing in it. Kept so a test can assert on
// the slip a wheel is working at rather than only on the force that came out of it.
export struct WheelContact
{
    glm::dvec3 forward{0.0, 0.0, 1.0};
    glm::dvec3 lateral{1.0, 0.0, 0.0};
    double longitudinalVelocity = 0.0;
    double lateralVelocity = 0.0;
    double effectiveRadius = 0.0;
    TyreSlip slip;
    TyreForces tyre;
};

// Everything the tick worked out about one corner, kept so the caller can fill telemetry and the
// tests can assert on the parts rather than only on the whole.
export struct CornerSolution
{
    SuspensionState suspension;
    ContactPatch patch;
    CornerForces forces;
    WheelContact contact;
    double generalisedForce = 0.0;
    double generalisedInertia = 0.0;
    double damperVelocity = 0.0;

    // --- the linkage's range constraint ---------------------------------------------------------
    //
    // Which authored limit held this corner this tick, if either did, and the generalised impulse
    // the constraint applied to do it, N·m·s, positive toward bump; `forces.rangeLimit` is the same
    // impulse as a per-tick equivalent force at the wheel. `None` and 0.0 whenever the corner's own
    // velocity step lands inside the range — which, on the shipped Golf, is every tick of every
    // characterisation scenario; the settle drop every fixture starts with hangs all four wheels on
    // their droop limits through the bounce and is where these read something.
    RangeLimit rangeLimit = RangeLimit::None;
    double rangeImpulse = 0.0;

    // --- the chassis's acceleration in the corner's equation ------------------------------------
    //
    // The acceleration of the body-fixed point under this wheel centre that the velocity step
    // solved against, world frame, m/s² (2026-09-08 latest of all, docs/frame-acceleration-brief.md):
    // the chassis's linear acceleration, the angular acceleration's and the centripetal term's
    // share at the wheel's arm, and the four corners' reactions of this tick. It is what the chassis
    // then does, to rounding, on any tick the bodywork touched nothing and `drivelineReaction` is
    // off. Zero with `VehicleSetup::frameAcceleration` off; `frameSweeps` is how many passes the
    // coupled solve took to settle, 0 with it off.
    glm::dvec3 attachmentAcceleration{0.0};
    std::uint32_t frameSweeps = 0;

    // `dI/dq` of the corner's generalised inertia at the tick's position, kg·m²/rad — `2·m_u·C'·C''` —
    // differenced over ±`inertiaSlopeStep` from the solve the inertia is read from
    // (`cornerInertiaSlope`; docs/nonlinear-geometry-brief.md). 0 with `nonlinearGeometry` off.
    double inertiaSlope = 0.0;

    // --- the chassis's side of the unsprung mass's acceleration ---------------------------------
    //
    // (2026-09-08 latest of all (d), docs/chassis-kinematic-reaction-brief.md.) The force the
    // chassis received at this wheel centre beyond `−m_u·q̈·R·C'` — the Coriolis, centripetal and
    // design-offset parts of the unsprung mass's acceleration — and the couple that puts the
    // design-point mass's inertial force back at its own point, both world frame; and the whole
    // force the corner handed the chassis at the wheel centre, `−m_u·q̈·R·C'` included. Zero with
    // `VehicleSetup::kinematicReaction` off (the last one then carries `−m_u·q̈·R·C'` alone).
    glm::dvec3 kinematicForce{0.0};
    glm::dvec3 kinematicCouple{0.0};
    glm::dvec3 chassisReaction{0.0};

    // --- the kerb-contact path -----------------------------------------------------------------
    //
    // What the cylinder query kept for this wheel this tick, after the exclusion rule and the
    // per-face merge; `obstacleCount` of the array is live. Empty on a car with `kerbContact` off
    // and on every flat-road tick with it on.
    std::array<ObstacleContact, maxObstacleContacts> obstacles{};
    std::uint32_t obstacleCount = 0;

    // How many raw per-triangle contacts the query reported before the rule ran — the flat road
    // under the tyre included. The inertness fixture's positive control: a tick with candidates and
    // no survivors is the rule doing its job, where a tick with neither is a query that saw nothing.
    std::uint32_t obstacleCandidates = 0;

    // The normal force those contacts delivered, summed, newtons; the elevation of the largest
    // one's axis above the horizon, radians; and the torque their friction put on the wheel's spin
    // about the `upright · (+1, 0, 0)` axis, newton metres — which is how a driven wheel climbs a
    // kerb from rest. The first two are the trace's "did it engage".
    double obstacleNormalForce = 0.0;
    double obstacleAxisElevation = 0.0;
    double obstacleSpinTorque = 0.0;
};

// How high the car's floor is above the road, per corner and as the two figures a rake-sensitive
// map is looked up against.
//
// **The datum is `VehicleSetup::rideHeightReference`**, which the car states — see there for why it
// is stated and not read off the collision box. Each corner's is measured perpendicular to that
// corner's *own* contact patch, so a car on a banked road reports the clearance it actually has
// rather than a world-vertical drop.
//
// The version this replaces was `state.chassis.position.y - patch.centre.y` averaged over the four
// corners and then discarded — three separate faults in one expression. The centre of mass is not
// the floor, so it measured about 0.5 m where a ride height is about 0.12; a world-vertical
// difference is not a clearance on any road that is not level; and averaging four corners into one
// scalar destroys rake, which is the entire difference a rake-sensitive map exists to express.
//
// Nothing consumes this yet, and that is deliberate: the aero it is for is out of scope. What was
// not acceptable was leaving a *wrong* seam in place for that work to be built on.
export struct RideHeight
{
    std::array<double, cornerCount> corners{};

    // The axle means, and the difference between them. Positive rake is the rear sitting higher than
    // the front, which is the sense every aero map is drawn in.
    double front = 0.0;
    double rear = 0.0;
    double rake = 0.0;

    // How many corners were touching. A corner in the air is measured against the plane the others
    // define, which is the best that can be said about it; with none touching there is no road to
    // measure from and every figure above is zero.
    std::uint32_t grounded = 0;
};

export struct VehicleStep
{
    std::array<CornerSolution, cornerCount> corners{};
    RideHeight rideHeight;
    // What the bodywork was touching this tick, with the impulses the solver settled on. Kept so a
    // test can assert on the contact rather than only on where the car ended up.
    ContactManifold contacts;
    TelemetryFrame telemetry;
};

// What the road put on each wheel this tick — the tire's own reaction, from the one place it is
// computed. Positive drives the wheel forward, so a rolling wheel's is negative.
//
// The driveline needs it, and passing it rather than letting the driveline guess is the same rule
// `wheelInertias` follows. A coupling's constraint torque is `(Id*Te - Ie*Td)/(Ie+Id)`, and in first
// gear the engine's reflected inertia is twelve times the wheels', so `Td` is most of the answer: the
// term the engine's own torque contributes is 7.6% of it. Told `Td = 0`, the only thing left in the
// constraint's arithmetic that can make up the difference is a speed difference, and a *locked*
// clutch then carries a steady slip — measured at 18 rad/s in first gear, 170 rpm of engine speed
// that should not be there.
export [[nodiscard]] std::array<double, cornerCount> roadTorques(const VehicleStep& step);

// The spring free length that puts a corner in equilibrium at its design position under a given
// sprung load. Not required — a car settles wherever its springs put it — but a setup whose design
// position is also its static position is the one whose camber and roll centre curves mean what
// they say, and computing it beats guessing at it. Solved on the **spring's own element**
// (`springElementOf`) since the spring migration; on a coaxial car that is the damper's element
// and the answer is bit-identical to what it always was.
export [[nodiscard]] std::expected<double, std::string> springFreeLengthForLoad(const CornerSetup& corner,
                                                                                const double sprungLoad);

// The spring's own solution at a solved suspension position: its element's length and Jacobian
// beside the force that length produces. This is what the force pass consumes since the spring
// migrated onto its own element — and the seam a test reaches the production spring arithmetic
// through without standing up a whole vehicle. Closed form, cannot fail.
export struct SpringSolution
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
    double force = 0.0;
};

export [[nodiscard]] SpringSolution solveSpringForce(const CornerSetup& corner, const SuspensionState& suspension);

// The damper's own geometry at a solved position, from the element evaluator — the damper-side
// counterpart of `SpringSolution`, and since step 7 the source of the production damper velocity.
// Deliberately geometry only: the damper force, the stops and the implicit damping still read the
// solver's state fields, and their migration is a later step's. On every current car the element
// is the solver's own damper, so length and Jacobian are the state's values bit for bit.
export struct DamperSolution
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
};

export [[nodiscard]] DamperSolution solveDamperGeometry(const CornerSetup& corner, const SuspensionState& suspension);

// The damper's force at a solved position and wishbone rate — the damper-side counterpart of
// `SpringSolution`'s force, and since step 8 what the force pass consumes. The force law is
// untouched: the same curve, the same compression-positive velocity `−lengthPerAngle · q̇`, the
// same signs; only the geometry source is the damper's own element. The stops and the implicit
// damping deliberately still read the solver's state fields.
export struct DamperForceSolution
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
    double velocity = 0.0;
    double force = 0.0;
};

export [[nodiscard]] DamperForceSolution solveDamperForce(const CornerSetup& corner, const SuspensionState& suspension,
                                                          const double wishboneRate);

// The damper's contribution to the corner's implicit damping, as a coefficient rather than a
// force: the damper's own Jacobian squared times the curve's local slope at the current shaft
// velocity, floored at zero. The law is untouched since before the element migrations — this is
// the same expression the integration always solved against, now stated on `DamperForceSolution`
// so the coefficient reads the damper's element and a test can hold it to the old bits.
export [[nodiscard]] double damperDampingCoefficient(const CornerSetup& corner, const DamperForceSolution& damper);

// --- the corner's velocity step ----------------------------------------------------------------
//
// One corner's one degree of freedom, stepped from a wishbone rate to the next (2026-09-08 later,
// docs/damper-integrator-brief.md). `generalisedForce` is the whole explicit generalised force at
// `previousRate` — the damper's own force at that rate included, as the force pass has always
// assembled it — `damperCoefficient` is that damper's floored local slope on the corner's Jacobian
// (`damperDampingCoefficient`), and `stopCoefficient` an engaged stop's viscous coefficient on the
// same Jacobian, or zero.
//
// The step is one Newton iteration of backward Euler on the damper, taken from the old rate:
//
//     G_d(q̇) ≈ G_d(q̇₀) − C·(q̇ − q̇₀)
//     q̇₁ = [ q̇₀ + (G + C·q̇₀)·dt/I ] / [ 1 + (C + C_s)·dt/I ]
//
// so the numerator carries the damper's affine intercept, `G_d(q̇₀) − G_d'(q̇₀)·q̇₀` — which for a law
// linear through the origin is exactly zero — and the divisor carries its slope, once. An engaged
// stop's viscous term is linear in the rate at a fixed compression and has no intercept, so it sits
// in the divisor alone, which is the form docs/stop-element-brief.md shipped. Before this the
// numerator carried `G_d(q̇₀)` whole beside the divisor's slope: for `c·v` that map is
// `(1 − β)/(1 + β)` where backward Euler's is `1/(1 + β)`, every damper slope acted about twice, and
// past `β = 1` the velocity changed sign every tick.
//
// With `damperCoefficient` zero the arithmetic is character for character the step this corner has
// always taken, which is what makes a car with no damper the byte-inert control.
export struct CornerRateStep
{
    double previousRate = 0.0;       // q̇₀, rad/s
    double generalisedForce = 0.0;   // G at q̇₀, N·m — the damper's own force at q̇₀ included
    double generalisedInertia = 0.0; // I, kg·m²
    double damperCoefficient = 0.0;  // C = J²·max(0, F_d'(v₀)), N·m·s
    double stopCoefficient = 0.0;    // C_s, the engaged stop's, N·m·s; zero when none is
    double deltaTime = 0.0;          // s
};

export [[nodiscard]] double solveCornerRate(const CornerRateStep& step);

// --- the linkage's range -----------------------------------------------------------------------
//
// The authored range `[droopAngle, bumpAngle]` as a unilateral constraint on the corner's coordinate
// (2026-09-08 later still, docs/range-constraint-brief.md). Taken after the velocity step and before
// the implicit shares are read off the rate. The step's own rate is kept whenever it lands the
// coordinate inside the range — the bits `solveCornerRate` produced, untouched — and is replaced by
// the one rate that lands it exactly on the limit it would have crossed, `(limit − q₀)/dt`, when it
// does not. A corner already resting on its limit therefore reads a rate of exactly zero for as long
// as the free step points outward, and leaves the moment the free step points inward, because a free
// step that points inward lands inside and is kept as it is. That is the complementarity condition,
// stated on the position the step would reach: the constraint acts only to stop a crossing, and it
// can never pull.
//
// It is a constraint and not a force law. Nothing here is sized from a penetration or a stiffness;
// the impulse that acted is whatever the corner's discrete acceleration then says it was —
// `I·(q̇₁ − q̇₀)/dt` less every other generalised force at the constrained rate — which the force pass
// folds into the generalised force the chassis reaction reads, so the body receives the equal and
// opposite reaction through the same Jacobian every other corner force does, and the damper's and
// the stop's implicit shares are read off the rate that acts rather than the one that was thrown
// away. Before this the range was a clamp on the state applied after the chassis had been given the
// reaction of the unconstrained acceleration: the body was pushed up by a hanging wheel that was in
// fact pulling it down, the damper reported the force of a shaft velocity the clamp then discarded,
// and the wheel's momentum went nowhere.
export struct RangeConstraintStep
{
    double previousAngle = 0.0; // q₀, rad — the coordinate the tick started from
    double rate = 0.0;          // q̇₁, rad/s — the velocity step's own rate
    double droopAngle = 0.0;    // the authored range, rad
    double bumpAngle = 0.0;
    double deltaTime = 0.0;     // s
};

export struct RangeConstraintSolution
{
    double rate = 0.0;                   // the rate that acts: the step's own, or the one that lands on the limit
    RangeLimit limit = RangeLimit::None; // which limit held, if either
};

export [[nodiscard]] RangeConstraintSolution constrainCornerRange(const RangeConstraintStep& step);

// --- the chassis's acceleration at a wheel's attachment -----------------------------------------
//
// (2026-09-08 latest of all, docs/frame-acceleration-brief.md.) What the rigid body's own mobility
// does at one wheel centre when a generalised force acts at another: the acceleration of the point
// at `armTo` along `jacobianTo` per unit force applied at `armFrom` along `jacobianFrom`, both arms
// from the centre of mass and both Jacobians in the world frame,
//
//     G = J_to · J_from / M + (r_to × J_to) · I⁻¹ · (r_from × J_from)
//
// symmetric in the two corners, and the whole of the chassis's response to a corner's reaction
// `−m_u·q̈·J` at that corner's own coordinate and at the other three. A corner's own entry is what
// turns its inertia into the reduced one, `m_u·|J|² − m_u²·G`.
export [[nodiscard]] double attachmentMobility(double mass, const glm::dmat3& worldInverseInertia,
                                               const glm::dvec3& armFrom, const glm::dvec3& jacobianFrom,
                                               const glm::dvec3& armTo, const glm::dvec3& jacobianTo);

// The acceleration of a body-fixed point at `arm` from the centre of mass of a body accelerating at
// `acceleration`, turning at `angularVelocity` and being spun up at `angularAcceleration`, all in the
// world frame: `a + α × r + ω × (ω × r)`. The Coriolis term of a point moving *in* the body frame is
// deliberately absent: it is perpendicular to the point's own relative velocity and drops out of
// the corner's equation, which projects onto that direction.
export struct ChassisMotion
{
    glm::dvec3 acceleration{0.0};
    glm::dvec3 angularVelocity{0.0};
    glm::dvec3 angularAcceleration{0.0};
};

export [[nodiscard]] glm::dvec3 attachmentAcceleration(const ChassisMotion& chassis, const glm::dvec3& arm);

// --- the corner's configuration-dependent inertia ----------------------------------------------
//
// (2026-09-08 latest of all (c), docs/nonlinear-geometry-brief.md.) The generalised inertia of a
// corner's one degree of freedom at a wishbone angle, kg·m², **unfloored**: `m_u·|C'|²` on the
// geometric path and `m_u·C'_y²` on the springs path, the corner solve's own statement evaluated by a
// fresh solve at that angle and rack travel. Fails where the linkage has no solution.
export [[nodiscard]] std::expected<double, std::string> cornerGeneralisedInertia(const CornerSetup& corner,
                                                                                 double wishboneAngle,
                                                                                 double rackTravel,
                                                                                 bool geometricLoadPath);

// The step the slope below is differenced over, radians: a milliradian, a thousand times the solve's
// own Jacobian step, because the slope is a difference of two differences and the solve's rounding
// — a few ulps of a metre — over a microradian squared would be most of the answer.
export inline constexpr double inertiaSlopeStep = 1e-3;

// `dI/dq` at a wishbone angle, kg·m²/rad: the central difference of `cornerGeneralisedInertia` over
// ±`inertiaSlopeStep`, one-sided against the angle's own inertia where one neighbour is outside the
// linkage's range, and exactly 0.0 where neither neighbour solves. Equal to `2·m_u·C'·C''` to the
// truncation of the step. On a corner whose `|C'|` is the same at every angle it is exactly 0.0.
export [[nodiscard]] double cornerInertiaSlope(const CornerSetup& corner, double wishboneAngle, double rackTravel,
                                               bool geometricLoadPath);

// `C''(q)`, the second derivative of the wheel centre's position with the wishbone angle, body frame,
// m/rad² (2026-09-08 latest of all (d), docs/chassis-kinematic-reaction-brief.md): the central
// difference of the solve's own Jacobian over ±`inertiaSlopeStep`, from the same pair of solves the
// inertia slope is read from, so that `2·m_u·C'·C''` and `cornerInertiaSlope` are one geometry. The
// vertical component alone on the springs path, which is that path's statement of the Jacobian.
// Exactly zero where neither neighbour solves.
export [[nodiscard]] glm::dvec3 cornerJacobianCurvature(const CornerSetup& corner, double wishboneAngle,
                                                        double rackTravel, bool geometricLoadPath);

// --- the chassis's side of the unsprung mass's acceleration ------------------------------------
//
// (2026-09-08 latest of all (d), docs/chassis-kinematic-reaction-brief.md.) What one unsprung mass
// asks of the chassis beyond the reaction of its own coordinate's acceleration. The chassis body
// carries the mass rigidly at its design wheel centre, so the true momentum of the mass differs from
// the body's by `m_u·(ẋ_u − ẋ_design)`, and the rate of that difference less the `m_u·R·C'·q̈` the
// force pass already applies is
//
//     F = −m_u·[ α × Δr + ω × (ω × Δr) + 2·ω × (R·C')·q̇ + (R·C'')·q̇² ]
//
// at the wheel centre; the angular momentum about the origin asks in addition that the design point's
// own inertial force, which the body's tensor carries at the design point, be taken back there —
// a couple `−m_u·Δr × (a_design − g)`, `a_design = a + α × r_design + ω × (ω × r_design)`. All in the
// world frame; `arm` and `designArm` from the centre of mass; `jacobian` and `curvature` `R·C'` and
// `R·C''`; `rate` the coordinate's rate. Even in the rate where the rate appears squared, odd in it
// where it appears once, and exactly zero with the offset, the rates and the curvature all zero.
export struct UnsprungKinematics
{
    double mass = 0.0;
    glm::dvec3 jacobian{0.0};
    glm::dvec3 curvature{0.0};
    glm::dvec3 arm{0.0};
    glm::dvec3 designArm{0.0};
    double rate = 0.0;
};

export struct KinematicReaction
{
    glm::dvec3 force{0.0};
    glm::dvec3 couple{0.0};
};

export [[nodiscard]] KinematicReaction unsprungKinematicReaction(const UnsprungKinematics& wheel,
                                                                 const ChassisMotion& chassis,
                                                                 const glm::dvec3& gravity);

// --- damper friction against shaft speed -------------------------------------------------------
//
// The shaft speed `CornerSetup::damperFriction` is quoted at, metres per second: **0.5 mm/s**, which
// is the source's own definition of quasi-static sliding friction — *"Quasi-static friction
// represents sliding friction at low velocities (typically at 0.5 mm/s)"* — and the slowest velocity
// its steady-state programme measures. A normalised shape is 1.0 here, so the magnitude keeps
// meaning exactly what the measurement it came from means.
export inline constexpr double damperFrictionReferenceSpeed = 0.0005;

// The normalised velocity factor a corner's stated shape gives at a shaft speed, metres per second,
// **speed and not velocity** — the sign is the regularised `tanh`'s job and this must not carry one.
// A corner that states no shape gets exactly 1.0, which is what makes the shipped law bit-identical.
export [[nodiscard]] double damperFrictionShapeAt(const Curve& shape, double speed);

// **A candidate velocity shape for a MacPherson front strut, and the one thing in this file sourced
// from a steady-state friction measurement.**
//
// Deubel, Dittrich, Meinck and Prokop, *Experimental analysis and modelling of friction in a
// MacPherson strut shock absorber under side load*, Tribology International **215 (2026) 111328**,
// their Fig. 4 — steady-state Stribeck curves of a **VW Passat B8 front strut**, the same specimen
// and the same paper the shipped 107 N comes from. Read off the design-deflection panel (`K0_SA`,
// the middle of five) at the side force whose quasi-static value **is** that 107 N, which is about
// 500 N. Normalised by its own 0.5 mm/s value, so this is a shape and not a second magnitude.
//
// The shape it states, and none of it is smoothed: friction **rises** from the quasi-static value to
// a maximum of about 1.31× it near 5 mm/s, then **falls monotonically** to 0.69× at the source's
// 300 mm/s cap. The paper's own words for why that is not the textbook Stribeck curve: it *"differs
// from the typical Stribeck curve in the sense that there is no higher static or boundary friction
// compared to higher velocities"*.
//
// **The grades, stated first.** (1) Every knot is read off a printed figure, ±5 N on a separated
// line and worse through the near-vertical rise, whose line width alone is about 7 mm/s — so the
// peak's *value* is well determined and its *velocity* is bounded only to 1–10 mm/s. (2) The shape
// is not invariant across the figure: at 250 N of side force the peak is 1.45× and the 300 mm/s tail
// 0.85×, at 750 N they are 1.22× and 0.67×, so picking the column is picking a cornering condition.
// (3) It is a **front strut** shape and there is no rear equivalent published anywhere reachable —
// see `docs/suspension-fidelity-brief.md`, the 2026-09-05 entry, for the fetch log.
//
// **Nothing installs it.** No car in this project states a shape; `front.frictionshape 1` on a setup
// sheet installs this on that axle for one session, and no seat verdict exists on it.
export [[nodiscard]] Curve macPhersonStrutFrictionShape();

// The anti-roll bar at one corner: its force, and the Jacobian that force rides.
//
// Two models behind one function, chosen by the data (`dropLinkStated`). A corner with no drop-link
// hardpoints gets the wheel-referred bar this model has always had — a rate times the difference in
// *wheel travel*, on the wheel's own vertical Jacobian, arithmetic for arithmetic. A corner that
// states one gets the bar on its own element: the rate referred to the link through the link's design
// motion ratio, the force from the difference in *link* displacement, and the projection the link's
// own. Both report a wheel-equivalent force so that `CornerForces::antiRoll` means one thing.
//
// The referral uses **both** corners' design ratios, so the pair's link forces are exactly equal and
// opposite even if an axle is not mirrored — which is a stronger statement than the wheel-referred
// model could make.
export struct AntiRollBarSolution
{
    // Along the drop link, newtons, positive pushing its ends apart. Zero without a stated link.
    double linkForce = 0.0;
    // d(link length)/dWishboneAngle, the Jacobian `linkForce` acts through. Zero without one.
    double lengthPerAngle = 0.0;
    // The equivalent force at the wheel, which is what the corner's generalised force uses in the
    // wheel-referred model and what both models report.
    double wheelForce = 0.0;
    // Which of the two models answered.
    bool geometric = false;
};

export [[nodiscard]] AntiRollBarSolution solveAntiRollBar(const CornerSetup& corner, const SuspensionState& suspension,
                                                          const CornerSetup& across,
                                                          const SuspensionState& acrossSuspension);

// The damper shaft's displacement from its design length, positive in bump — the whole of what
// the bump and droop stops measure their gaps against, because a real stop lives on the shaft.
// Since step 12 both lengths are the damper element's: the design length evaluated at q = 0 and
// the current length off the same element, so the stops' kinematics follow the damper wherever
// its geometry goes. The stop force law and the stops' generalised contribution are untouched.
export [[nodiscard]] double damperShaftCompression(const CornerSetup& corner, const DamperForceSolution& damper);

// Standard gravity, and the only constant in this file that is not a placeholder.
inline constexpr auto earthGravity = 9.80665;

// What the outside front tyre carries when this car is cornering at its own limit, newtons.
//
// **A fixed point rather than a chosen manoeuvre**, which is what makes it a property of the car
// instead of a number somebody drove to. Lateral load transfer is `m_axle · a_y · h / t`, and the
// lateral acceleration the car can hold is set by the friction the outside tyre has — which falls
// with the very load the transfer is putting on it. So the two define each other, and the honest
// statement is the load that satisfies both:
//
//     Fz = W_front/2 + mu(Fz) · m_front · g · h / t_front
//
// It converges in three or four passes from the static load because tyre load sensitivity is weak
// (this car's exponent is 0.1926) and the map is a strong contraction.
//
// **Validated against a real lap.** For the Golf this returns 6375 N on the outside front, against a
// median 5851 N measured on Dominic's 201-second Bathurst session through the slip band where the
// aligning moment peaks — **9% high**, and high is the expected direction: the approximation here is
// that the front axle carries its own transfer with the whole car's centre of gravity height, rather
// than a share set by the front/rear roll-stiffness split, and it takes no account of a driver who
// is not at the limit on every corner. What is being placed against it is a knee and a taper, so
// nine percent is comfortably inside what the answer needs to be worth.
//
// The rack force that comes out the far end of it agrees better than the load does: 1743 N derived
// against 1617 N measured, 7.8%, because the geometry between them is exact and only the load is
// modelled.
export struct SteeringLimitLoads
{
    // What one front wheel carries with the car at rest, newtons.
    double staticPerWheel = 0.0;
    // And what the outside and inside front carry at the limit. They sum to twice the static one:
    // lateral transfer moves load across the axle, it does not add any.
    double outside = 0.0;
    double inside = 0.0;
};

export [[nodiscard]] SteeringLimitLoads steeringLimitLoad(const VehicleSetup& setup);

// Placeholder geometry for one corner, mirrored by side and placed at an axle station. Short-long
// arm, so camber gain comes out of the linkage rather than being asked for.
export [[nodiscard]] CornerHardpoints placeholderCorner(const CornerSide side, const double axleZ);

// Refuses a corner whose stops cannot do their job, which is a different question from whether its
// geometry is sound and is not covered by `validateCorner`.
//
// The failure this exists for is worth stating, because it does not look like a suspension problem
// at all. A stop whose gap is larger than the shaft's travel never engages; the corner runs instead
// to the linkage's authored limit. **Until 2026-09-08 later still that limit was a clamp with no
// reaction force**: the spring went on pushing the chassis up with nothing pushing back, so a car in
// the air gained energy every tick, pitched onto its nose and eventually left the world, and it read
// as an integrator fault or a contact fault and was neither. The limit is a unilateral constraint
// now (`constrainCornerRange`) and carries whatever the stop cannot; the refusal stays because a
// stop that never engages is still an authoring mistake, and this is where it is caught.
export [[nodiscard]] std::expected<void, std::string> validateCornerSetup(const CornerSetup& corner);

// **A placeholder car**, and every number in it is one: a 1352 kg mid-size sedan, 2.7 m wheelbase,
// 1.44 m track, 60/40 front weight distribution, ride frequencies of about 1.2 Hz front and 1.3 Hz
// rear. They are documented real-world figures for the class rather than invented ones, so the model
// behaves plausibly while it is being validated, and swapping in a measured car is a change to this
// function and to nothing else.
export [[nodiscard]] std::expected<VehicleSetup, std::string> placeholderSedan();

// One tick of one vehicle. Pure in (setup, state, input, driveTorques, dt) and the world it is
// standing on: no clock is read, no global is touched, nothing is random. The structure is the
// architecture the brief asks for and everything later plugs into it — clear the accumulators, let
// every element contribute a force, then integrate once at the bottom. No element writes a velocity
// or a position.
//
// `driveTorques` is what the driveline is putting on each wheel, in newton metres, and it arrives as
// its own argument rather than as a field on `VehicleInput` deliberately. `VehicleInput` is the
// driver's — steering, throttle, brake, gear — and is the packet a rollback netcode would transmit
// and replay. Driveline torque is *derived* from state and must be recomputed on a replay rather
// than trusted from the wire, so it has no business in the struct that gets sent.
//
// `brakes` is the assist layer's output and follows the same rule for the same reason. Defaulted,
// and the default is not "no brakes" — it is **"nobody intervened"**, under which this function
// applies the driver's demand against each corner's own peak exactly as it did before an assist
// layer existed. That is what lets every fixture in the suite go on calling this with six arguments
// and get an answer identical to the bit rather than merely close, which is the only kind of
// regression evidence worth having for a car underneath a controller.
//
// `ambient` is the weather, and it follows the same rule for the same reason: it is a property of
// the *scene* rather than of the car, it is read by exactly one thing — the tread's heat balance —
// and it is defaulted so that every fixture that does not care goes on calling this with six or
// seven arguments. A car with `tyreThermal` off never reads it at all.
//
// `obstacles` is whatever moving solids the *caller* owns that this car might hit this tick — traffic.
// They are not in the physics world and cannot be: that world is immutable once created, which is
// what makes it safe to query from the simulation thread. So they arrive here, are collided against
// the bodywork's own box, and are resolved in the same sequential-impulse pass as the road and the
// buildings rather than in a second one that would undo the first. What the solver gave each of them
// comes back in `VehicleStep::contacts.bodies`, keyed by `ContactBody::obstacle`.
//
// **Defaulted to nothing, and an empty span is bit-identical to the car that shipped before traffic
// existed** — `collideObstacles` returns before touching the manifold. Both frame gates pin a
// circuit with no traffic lanes, so both are blind to the whole of this by construction.
export [[nodiscard]] std::expected<VehicleStep, std::string>
stepVehicle(const VehicleSetup& setup, VehicleState& state, const VehicleInput& input,
            const std::array<double, cornerCount>& driveTorques, const PhysicsWorld& world, const double deltaTime,
            const BrakeCommand& brakes = {}, const AmbientConditions& ambient = {},
            std::span<const DynamicObstacle> dynamicObstacles = {});

} // namespace raceengine
