module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:ContactPatch;

import :PhysicsWorld;
import :ProvingGround;

namespace raceengine
{

// Where the road actually is, under a wheel, as more than one number.
//
// A single ray at the wheel centre reads adequately while a car is tracking a lane and is wrong the
// moment anything interesting happens — a wheel half on a kerb, straddling a tarmac/grass edge, or
// landing at an angle — which in this game is most of the time. Its failure is specific and it is
// not "slightly inaccurate": one ray reports a uniform lift where the truth is a *tilted* patch
// carrying more load on one side, so the moment that should roll the car never appears at all.
//
// So the wheel asks a grid of questions and aggregates the answers into an effective plane, a
// penetration depth, and a patch centre that moves. The migration is the point: a wheel with one
// shoulder on a kerb has its load centred outboard of the wheel's middle, and tire forces applied
// there produce the right moment without anything modelling that moment explicitly.

export struct ContactPatchSampling
{
    // **Seven across and three along, and the asymmetry is the point.** A kerb is a longitudinal
    // feature crossed transversely, so the resolution is wanted *across* the patch and there is
    // nothing happening along it worth resolving. Seven across also puts the elements 33 mm apart,
    // which is the coarsest grid that resolves a belt of the length this tyre is given at all — at
    // three across they sit 100 mm apart and the coupling is worth 0.04 of an element's radial rate,
    // which is the uncoupled bed wearing a different name.
    //
    // It costs. The sampler is 93% raycast and linear in rays, so this is 84 rays a car against 36,
    // and with the belt solved it takes the contact sampling from about a third of the vehicle tick's
    // 50 microsecond budget to four fifths of it. Seven-by-three is deliberately not five-by-five:
    // it buys more lateral resolution for less cost, which is the row that decided it.
    std::uint32_t across = 7;
    std::uint32_t along = 3;

    // Patch dimensions, in metres, and **parameters rather than constants** — a pressure or thermal
    // model changes the size of the contact patch, which changes the extent this grid has to cover.
    // Nothing varies them this milestone; the seam is what matters.
    double width = 0.20;
    double length = 0.16;

    // How far above the undeformed tire surface each ray starts, and how far it looks. Generous:
    // the cost of a longer ray is nothing next to the cost of a wheel dropping through the world
    // because the road was further away than the search.
    double searchDistance = 1.0;

    // A sample penetrating this much deeper than the patch's median is treated as a defect in the
    // collision mesh rather than as road. A single triangle out of place, or a seam between two of
    // them, otherwise carries the whole wheel: load share goes as penetration, so the deepest
    // sample dominates exactly when it is least trustworthy.
    //
    // **The comment that used to sit here said fifty millimetres is "well past any real kerb this
    // grid would straddle", and the proving ground's own `kerbHeight` is exactly fifty.** It was
    // wrong, and it was also the recorded cause of the kerb-edge load dip, which it is not: measured
    // across the whole chamfer crossing, at every sample count from 3x3 to 31x31, **zero samples are
    // ever rejected** (`./EngineTests "[.patch-chamfer]"`). It cannot fire on this feature by
    // construction either — a 3x3 grid spans about 0.25 m of a 0.30 m chamfer, so the depth spread
    // it can see across a 9.5-degree ramp is bounded at about 42 mm.
    //
    // The number stays what it was, because what it is *for* is a mesh fault, and a mesh fault is
    // not a kerb. What is gone is the claim that a kerb could not reach it.
    double spikeRejection = 0.05;

    // How far along the belt a load spreads: the distance over which one element's deflection decays
    // into its neighbours' — the carcass's radial bending, expressed as a length. **Zero is not a
    // degenerate setting, it is the bed of independent springs this model replaced**, and it is
    // exactly reproduced: every quantity the aggregate reports comes back bit-identical at zero, which
    // is what makes the change provable rather than merely plausible.
    //
    // It is *not* `TyreModel::lateralRelaxation` or its longitudinal partner, which are already
    // stated at 0.50 m and 0.30 m. Those are in-plane shear lags — how far the tyre rolls before a
    // slip force builds — and this is radial bending across the belt's width. Sharing a unit does not
    // make them the same length, and using one for the other would be off by an order of magnitude.
    //
    // See `solveBeltBed` for what the number does and `docs/tyre-enveloping-and-dsg-brief.md` for
    // how it was set.
    double beltBridgingLength = 0.060;

    // Two adjacent samples whose road normals differ by more than this are not on the same piece of
    // road, and the belt between them bends rather than shears. The coupling is cut there.
    //
    // Five degrees, and the number is a measurement rather than a margin: surveyed over
    // `rt_bathurst_physics.glb`, the worst adjacent pair inside a patch has a median of 0.082 degrees
    // on the racing line and 4.588 over kerb triangles, and 0.00% of tarmac patches against 47.5% of
    // kerb patches clear five. That is about 600:1, and the overlap is a p99.9 of 4.68 degrees on
    // tarmac swept four metres either side — road edges and camber transitions, where cutting is
    // right. `scripts/mesh-normal-survey.py` is the survey.
    double beltBreakAngle = 0.08726646259971647;

    // Sweeps of projected Gauss-Seidel per patch per tick. **Fixed, and never an early-out on a
    // convergence test**: both parity gates are byte-identical and a solver that stops when it is
    // happy makes the frame a function of the arithmetic's mood. Scoped in `EnvelopeSolverProbe.cpp`.
    std::uint32_t beltIterations = 32;
};

// The wheel as the contact sampler needs it: a cylinder of `radius` about `spinAxis`, centred at
// `centre`. `forward` fixes which way the patch's long axis points and is orthogonalised against
// the spin axis on the way in, so a caller passing the upright's forward direction cannot make the
// two disagree.
export struct WheelPose
{
    glm::dvec3 centre{0.0};
    glm::dvec3 spinAxis{1.0, 0.0, 0.0};
    glm::dvec3 forward{0.0, 0.0, 1.0};
    double radius = 0.31;
};

// The rays for one wheel, and where the undeformed tire sits under each of them. Produced apart
// from the aggregation so that every wheel's rays can go into one batched cast — which is what the
// world's query interface is shaped for — and so that both halves stay pure and testable without a
// device.
export struct ContactSampleGeometry
{
    std::vector<glm::dvec3> origins;
    std::vector<glm::dvec3> directions;
    // The point on the *undeformed* tire directly under each sample. Penetration is measured
    // against these, so the tire's own shape is accounted for before the road is.
    std::vector<glm::dvec3> tireSurface;
};

export struct ContactPatch
{
    bool inContact = false;

    // Load-weighted, all three: the aggregate leans toward the samples actually carrying the wheel,
    // which is what tilts the plane and moves the centre when one shoulder is on a kerb.
    glm::dvec3 centre{0.0};
    glm::dvec3 normal{0.0, 1.0, 0.0};
    double gripMultiplier = 1.0;
    double bumpiness = 0.0;

    // The mean **belt deflection** over the whole grid, including the elements touching nothing —
    // which is a different aggregate from the three above and deliberately so. Vertical force goes as
    // the total compression across the patch, so a wheel with half its patch in the air must carry
    // about half the load. Weighting this one by load share too would report the deep half's
    // penetration and hand back a wheel carrying full load on half a contact patch.
    //
    // **It is the deflection and not the depth** (2026-08-22, E2). On the bed of independent springs
    // this replaced the two were the same number: an element deflected exactly as far as the road
    // pushed it and an element the road had left behind deflected not at all. With the belt coupled,
    // an element over a dip is dragged inward by its loaded neighbours and its radial spring carries
    // a share of the load, which is the whole of what enveloping is. The sum is still exactly the sum
    // of the contact pressures — see `solveBeltBed` for why that cancellation is exact — so this is
    // still the number the tyre's vertical rate multiplies, and it is still bounded by the deepest
    // element rather than invented.
    //
    // It is a **quadrature, and it converges rather than being exact**: even on perfectly flat
    // ground the compression varies across the patch because the tread curves away from it, so a
    // 3x3 grid and a 7x7 grid over the same wheel disagree. That is not an error to be fixed — it is
    // the accuracy of three points across a curve — but it does mean the sample count is part of a
    // vehicle's configuration rather than free to change under a car that has already been given a
    // spring rate. The normal, the grip and the patch centre have no such dependence.
    //
    // Measured 2026-08-21 at a held 15 mm of peak compression, the disagreement is larger than the
    // 15% recorded here before anyone ran it: flat ground reads **2391 N at 3x3 against 3375 N at
    // 31x31**, still climbing, and it is *the same* 41% at every position on flat road either side
    // of a kerb. It is a scale factor on the vertical rate, absorbed into whatever spring the car
    // was then given, which is precisely why the sample count is not free.
    //
    // **The kerb-edge load dip is measured in this aggregate and is largely not in it** (corrected
    // 2026-08-22, E2). The recorded finding is that a wheel held at 15 mm of *deepest* compression on
    // a kerb chamfer reads 898 N where flat ground reads 2450, and that refining the grid deepens it
    // rather than closing it — both still true, and pinned by the characterisation test next door.
    //
    // What is not true is the conclusion drawn from it. Walk the same wheel across the same kerb at a
    // held *ride height*, which is the control a car on springs actually has, and there is no dip at
    // the chamfer at all: the load goes 4061 N on the flat, 5441 at the inner edge and 4322 through
    // the middle of the ramp. Holding the deepest sample fixed puts the wheel high enough that only
    // its up-slope shoulder is on the road — the road under the middle of its patch is fourteen
    // millimetres *below* the tread — so a low reading there is the placement and not the model.
    // That control was chosen for a convergence study, where it is the right one, and then read as a
    // physical claim.
    //
    // The coupled bed was built and measured against it (`solveBeltBed`, `./EngineTests
    // "[.envelope-solver]"`) and does not remove it: at a 30 mm bridging length the chamfer reads
    // 813 N against 727 uncoupled, and at the converged grid the belt is worth about 15%. It is off
    // by default for that reason. What *is* real under the held-height control is the drop across the
    // kerb's top edge — 4322 N to 2941 as the patch straddles the crest — and how far a tyre should
    // bridge a crest is a number this project does not have.
    double penetration = 0.0;

    std::uint32_t contactingSamples = 0;
    std::uint32_t totalSamples = 0;

    // Deepest minus shallowest across the samples that carry load, in metres — the same set
    // `contactingSamples` counts, so the two are read together and mean the same thing by
    // construction.
    //
    // **It is here because `contactingSamples` alone cannot tell two opposite cases apart**, and
    // they need opposite treatment:
    //
    // - *many samples, wide spread* is road geometry under the patch — a chamfer, a kerb, a crest.
    //   The tyre is loaded, the belt is bridging the tilt, and the load should be roughly preserved
    //   on a smaller patch at higher pressure. This is the case the spring bed gets wrong.
    // - *few samples, narrow spread* is a shallow patch that is genuinely lightly loaded. The load
    //   should be low, and the model is already right.
    //
    // `Fz` cannot make that distinction, and this is the reason the channel exists rather than a
    // preference for more channels: `Fz` is computed from `penetration`, which is the aggregate the
    // defect corrupts. A loaded wheel on a kerb edge reports a *low* load **because of** the bug, so
    // any discriminator downstream of the aggregation classifies it as a light tyre and argues in a
    // circle. This one is taken from the per-sample depths before they are averaged, which is
    // upstream of the corruption and is the only place it can be taken from.
    //
    // Zero when nothing is touching, and zero — not undefined — when exactly one sample is: a
    // single sample has no spread, and a sentinel here would have to be special-cased by everything
    // that plots it.
    double depthSpread = 0.0;
};

// What the belt does under one patch, as more than a row of unconnected springs.
//
// **On by default at 60 mm since 2026-08-22, and the number came out of the seat rather than out of
// a measurement.** That provenance is the honest one and it is not a weakness: Dominic drove Bathurst
// at 0, 30, 60 and 120 mm and settled on 60, reporting that it "feels like iRacing". There is no cleat
// test or measured enveloping curve behind it, and there is none in this workspace to have. What the
// measurements did establish is that the model is *sound* — see the exact limits below — so what was
// left to choose was a length, and a driver against a reference simulator is a better instrument for
// that than anything the proving ground can offer.
//
// **What the measurements said, kept because it is still true and reads as an argument against.** The
// belt does not remove the kerb-edge load dip (that dip is mostly an artefact of the control it was
// measured under — see `ContactPatch::penetration`), and it makes a lateral kerb crossing *rougher* in
// the patch centre, five times at 30 mm. Both stand. The first turned out to be aimed at a defect that
// was never there, and the second was scored as a cost when the complaint being investigated was that
// kerbs felt too smooth — the same number with the sign read the wrong way round.
//
// **The mechanism.** A tyre asked to stand on a road that breaks or tilts inside its own contact patch
// cannot be a bed of independent radial springs: a Winkler bed has no bending stiffness, so nothing
// carries load across a cross-slope and every element that loses the road simply stops contributing.
// That is not a subtle modelling preference — it is why the tandem-cam models, the rigid ring with
// residual stiffness and FTire's flexible belt all exist. This is the cheapest member of that family
// that still never summarises the patch.
//
// **The one fidelity wrinkle worth knowing about, because it is not visible from the parameter.** The
// belt is given a single isotropic bridging length, but the grid is 7x3, so the elements sit 33 mm
// apart across the patch and 80 mm along it — and the coupling comes out about six times stiffer
// across than along. A real belt is stiffer the other way round, circumferentially, because that is
// where the steel runs. It is an artefact of the grid's aspect ratio rather than of the model, and it
// lands on the axis a kerb actually breaks, so it is recorded rather than corrected. Correcting it
// means a second stated length, and there is no data to set one with.
//
// Four repairs that summarise the patch with one global quantity were built, measured and refused
// first — a bridging rule against an effective plane, load-weighting, a kernel filter of the height
// field, and a fitted plane with a conformity limit. They fail for one reason, and it is the most
// useful sentence the investigation produced: **the defining feature of a kerb is a discontinuity
// *inside* the patch, so any summary of the patch destroys exactly the thing being asked about.**
// `docs/e2-enveloping-brief.md` carries each one and its measurement.
//
// So the coupling here is between *adjacent* elements and nothing else. It never forms a summary, it
// assumes nothing about the patch sitting on any one surface, and a break in slope inside the patch is
// just two elements with different depths and no shear term between them.
//
// **The model.** Element `i` sits at gap `g_i` — how far the road is above the undeformed tread — and
// deflects `w_i` inward. Each element has a radial spring to the rim, and adjacent elements are joined
// by a shear spring that resists their *difference*. The road can push and cannot pull, so
//
//     w_i >= g_i        the belt cannot be below the road
//     lambda_i >= 0     the road can only push
//     lambda_i * (w_i - g_i) = 0
//
// with `lambda_i = w_i + sum_j kappa_ij (w_i - w_j)` in units of the per-element radial rate. That is
// a linear complementarity problem, not a matrix inversion, and it is the reason this is solved rather
// than evaluated. It is also the *easy* kind: the matrix is symmetric, positive definite and an
// M-matrix, so it is a strictly convex quadratic with simple bounds and projected Gauss-Seidel
// converges monotonically on it.
//
// **Why the total load is still the sum of the deflections.** Summing every row,
//
//     sum_i lambda_i = sum_i w_i + sum_i sum_j kappa_ij (w_i - w_j) = sum_i w_i
//
// because each edge contributes `(w_i - w_j)` and `(w_j - w_i)` and they cancel. So the coupling
// **redistributes pressure and never invents load**, which is precisely what every refused candidate
// could not promise, and it is why `ContactPatch::penetration` keeps its old meaning and its old
// consumer: the mean deflection over the grid, times the tyre's vertical rate, is still the load.
//
// It also gives the model's two limits for free, and both are exact rather than approximate:
//
//   - **Every element in contact** — flat ground, a plane of any tilt the patch stays on — and the
//     sum is unchanged from the uncoupled bed, whatever the coupling. Shear moves pressure between
//     elements that are all pressing anyway, and the cancellation above is what says so.
//   - **Zero bridging length** and the whole thing reduces to `w_i = max(0, g_i)`, element by element,
//     which is the bed it replaced, bit for bit.
//
// What changes is the case in between: elements that have lost the road are dragged inward by the ones
// that have not, so they carry a share of the load instead of none. That share decays exponentially
// away from the contact, over the bridging length, which is what makes a genuine overhang still shed
// load rather than being bridged for free.
export struct BeltBed
{
    // Inward deflection per element, metres. Its mean over the whole grid is the penetration.
    std::vector<double> deflection;
    // Contact pressure per element, in units of the per-element radial rate — so it has the dimension
    // of a length, and summing it gives the same number as summing the deflections. Zero wherever the
    // element is not pressing on anything.
    std::vector<double> pressure;
};

// The shear rate between two elements a distance `spacing` apart, as a multiple of an element's radial
// rate, for a belt whose load spreads over `bridgingLength`.
//
// The **exact** discrete relation and not the continuum `(L/dx)^2` it tends to: an impulse on the
// infinite chain `w_i + kappa (2 w_i - w_{i-1} - w_{i+1}) = f_i` decays as `rho^n` with
// `rho + 1/rho = 2 + 1/kappa`, so asking for `rho = exp(-dx/L)` inverts to the expression below. Using
// it rather than the approximation is what makes the belt's response length *L* whatever the grid
// pitch is, instead of only in the limit — and the sample count is already part of a vehicle's
// configuration in this model, so one fewer thing depending on it is worth the cosh.
export [[nodiscard]] double beltShearRate(const double spacing, const double bridgingLength)
{
    if (!(bridgingLength > 0.0) || !(spacing > 0.0))
    {
        return 0.0;
    }

    // A grid far coarser than the belt's own response length cannot resolve it, and the honest answer
    // there is the uncoupled bed rather than a very small number obtained by overflowing a cosh.
    const auto ratio = spacing / bridgingLength;
    if (ratio > 40.0)
    {
        return 0.0;
    }

    return 1.0 / (2.0 * (std::cosh(ratio) - 1.0));
}

// The bed, solved. Pure, and deliberately reachable on its own: the convergence of a fixed iteration
// budget on a unilateral problem is the one part of this whose cost could not be argued from a
// measurement already taken, so it has to be measurable without a world, a wheel or a car.
//
// `gaps` and `constrained` are per element in row-major order — `along * across + across` — and an
// element that is not constrained is one the road is not touching: it has no lower bound and is free
// to be dragged wherever its neighbours put it. `cutAcross` and `cutAlong` mark edges the belt bends
// over rather than shears across, indexed by the *lower* of the two elements they join.
export [[nodiscard]] BeltBed solveBeltBed(const std::vector<double>& gaps, const std::vector<char>& constrained,
                                          const std::vector<char>& cutAcross, const std::vector<char>& cutAlong,
                                          const std::uint32_t across, const std::uint32_t along,
                                          const double shearAcross, const double shearAlong,
                                          const std::uint32_t iterations)
{
    const auto count = gaps.size();

    auto bed = BeltBed{};
    bed.deflection.assign(count, 0.0);
    bed.pressure.assign(count, 0.0);

    if (count == 0 || count != constrained.size() ||
        count != static_cast<std::size_t>(across) * static_cast<std::size_t>(along))
    {
        return bed;
    }

    // The uncoupled bed as the starting point. It is the exact answer when there is no coupling and a
    // good one when there is, so the iteration below starts inside the feasible set and stays there.
    for (auto index = std::size_t{0}; index < count; index++)
    {
        bed.deflection[index] = constrained[index] != 0 ? std::max(0.0, gaps[index]) : 0.0;
        bed.pressure[index] = bed.deflection[index];
    }

    // **Not an early-out on convergence, which is forbidden here.** Whether there is any coupling at
    // all is decided by the two shear rates before a single sweep runs and cannot depend on the
    // values, so this branch is taken identically on every machine and every run. With no coupling
    // the initialisation above is not an approximation, it is the answer — every row is `w = max(0,
    // g)` — and the sweeps would spend two microseconds a car re-deriving it.
    if (!(shearAcross > 0.0) && !(shearAlong > 0.0))
    {
        return bed;
    }

    // Every neighbour of an element, and what the belt between them is worth. Built once rather than
    // re-derived inside the sweep, because the sweep runs eight times over it.
    struct Neighbour
    {
        std::size_t other = 0;
        double shear = 0.0;
    };

    auto neighbours = std::vector<std::vector<Neighbour>>(count);
    auto diagonal = std::vector<double>(count, 1.0);

    const auto join = [&](const std::size_t a, const std::size_t b, const double shear)
    {
        if (!(shear > 0.0))
        {
            return;
        }

        neighbours[a].push_back(Neighbour{.other = b, .shear = shear});
        neighbours[b].push_back(Neighbour{.other = a, .shear = shear});
        diagonal[a] += shear;
        diagonal[b] += shear;
    };

    for (auto row = std::uint32_t{0}; row < along; row++)
    {
        for (auto column = std::uint32_t{0}; column + 1 < across; column++)
        {
            const auto left = static_cast<std::size_t>(row) * across + column;
            if (left < cutAcross.size() && cutAcross[left] == 0)
            {
                join(left, left + 1, shearAcross);
            }
        }
    }

    for (auto row = std::uint32_t{0}; row + 1 < along; row++)
    {
        for (auto column = std::uint32_t{0}; column < across; column++)
        {
            const auto near = static_cast<std::size_t>(row) * across + column;
            if (near < cutAlong.size() && cutAlong[near] == 0)
            {
                join(near, near + across, shearAlong);
            }
        }
    }

    // Projected Gauss-Seidel, a fixed number of sweeps and no early-out. Each row is solved as though
    // it carried no contact force and then pushed back up onto its own constraint, which is the
    // projection: an element the road is under cannot go below it, and one the neighbours lift clear
    // of the road simply stops pressing.
    for (auto sweep = std::uint32_t{0}; sweep < iterations; sweep++)
    {
        for (auto index = std::size_t{0}; index < count; index++)
        {
            auto pulled = 0.0;
            for (const auto& neighbour : neighbours[index])
            {
                pulled += neighbour.shear * bed.deflection[neighbour.other];
            }

            const auto free = pulled / diagonal[index];
            bed.deflection[index] = constrained[index] != 0 ? std::max(free, gaps[index]) : free;
        }
    }

    // The pressure each element ends up putting on the road. An element the road is not under presses
    // on nothing by construction and is written zero rather than evaluated — at exact convergence the
    // arithmetic would agree, and at a fixed budget it would hand a free element a phantom pressure
    // out of the residual. The rest is clamped for the same reason in the other direction: a
    // barely-loaded element can come out a few micronewtons negative, which has no meaning and would
    // subtract from a load-weighted mean.
    for (auto index = std::size_t{0}; index < count; index++)
    {
        // Cleared first and written second. The initialisation above seeds this with the uncoupled
        // answer so that the no-coupling path can return from there, and an element that turns out
        // not to be pressing has to lose that seed rather than keep it.
        bed.pressure[index] = 0.0;

        // **Whether the road is holding this element up is asked of the projection, not of the
        // arithmetic.** The sweep writes `max(free solve, gap)`, so an element the road is holding
        // carries *exactly* its gap and one the belt has lifted clear carries strictly more than it —
        // an exact test, and the only one that is safe at a fixed budget. Evaluating the row instead
        // would ask whether a lifted element's residual came out above or below zero, and Gauss-Seidel
        // leaves a residual by construction: the neighbours move again after this element was last
        // updated. That reads a lifted element as pressing, which puts it into the patch centre and
        // into the contacting count.
        if (constrained[index] == 0 || bed.deflection[index] > gaps[index])
        {
            continue;
        }

        auto shear = 0.0;
        for (const auto& neighbour : neighbours[index])
        {
            shear += neighbour.shear * (bed.deflection[index] - bed.deflection[neighbour.other]);
        }

        bed.pressure[index] = std::max(0.0, bed.deflection[index] + shear);
    }

    return bed;
}

namespace
{

[[nodiscard]] double offsetAt(const std::uint32_t index, const std::uint32_t count, const double extent)
{
    if (count < 2)
    {
        return 0.0;
    }

    return (static_cast<double>(index) / static_cast<double>(count - 1) - 0.5) * extent;
}

} // namespace

export [[nodiscard]] ContactSampleGeometry contactPatchSamples(const WheelPose& wheel,
                                                               const ContactPatchSampling& sampling)
{
    auto geometry = ContactSampleGeometry{};

    const auto spinAxis = glm::normalize(wheel.spinAxis);

    // The wheel plane's down direction. For a cambered wheel this is not the world's down, and that
    // difference is exactly what puts the contact patch inboard or outboard of the wheel centre.
    const auto worldDown = glm::dvec3(0.0, -1.0, 0.0);
    const auto inPlaneDown = glm::normalize(worldDown - glm::dot(worldDown, spinAxis) * spinAxis);

    // Orthogonalised rather than trusted, so the patch's axes are always a proper frame.
    auto forward = wheel.forward - glm::dot(wheel.forward, spinAxis) * spinAxis;
    forward = glm::length(forward) > 1e-9 ? glm::normalize(forward) : glm::cross(inPlaneDown, spinAxis);

    const auto count = static_cast<std::size_t>(sampling.across) * static_cast<std::size_t>(sampling.along);
    geometry.origins.reserve(count);
    geometry.directions.reserve(count);
    geometry.tireSurface.reserve(count);

    for (auto alongIndex = std::uint32_t{0}; alongIndex < sampling.along; alongIndex++)
    {
        const auto v = offsetAt(alongIndex, sampling.along, sampling.length);

        // The tire is a cylinder about the spin axis, so a sample ahead of or behind the patch's
        // middle sits on a shorter radius — the tread curves away. Ignoring this would report the
        // leading and trailing rows as penetrating less than they do and quietly shrink the patch.
        const auto drop = std::sqrt(std::max(0.0, wheel.radius * wheel.radius - v * v));

        for (auto acrossIndex = std::uint32_t{0}; acrossIndex < sampling.across; acrossIndex++)
        {
            const auto u = offsetAt(acrossIndex, sampling.across, sampling.width);

            const auto surface = wheel.centre + u * spinAxis + v * forward + drop * inPlaneDown;

            geometry.tireSurface.push_back(surface);
            geometry.origins.push_back(surface + glm::dvec3(0.0, sampling.searchDistance, 0.0));
            geometry.directions.push_back(worldDown);
        }
    }

    return geometry;
}

// The aggregate. Pure: it takes what the rays came back with and says where the road effectively
// is, so it can be tested against hand-written samples with no world at all.
export [[nodiscard]] ContactPatch aggregateContactPatch(const ContactSampleGeometry& geometry,
                                                        const std::vector<SurfaceHit>& hits,
                                                        const std::vector<SurfaceMaterial>& materials,
                                                        const ContactPatchSampling& sampling)
{
    auto patch = ContactPatch{};
    patch.totalSamples = static_cast<std::uint32_t>(hits.size());

    if (hits.empty() || hits.size() != geometry.tireSurface.size())
    {
        return patch;
    }

    // Vertical, not along the surface normal. The tire deflects towards the wheel centre and the
    // road pushes back along its own normal, but the *compression* of a vertically loaded tire is
    // the vertical overlap; measuring it along a steeply banked normal would report a wheel resting
    // on a slope as barely loaded.
    auto penetrations = std::vector<double>(hits.size(), 0.0);
    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        if (!hits[index].hit)
        {
            continue;
        }

        penetrations[index] = hits[index].point.y - geometry.tireSurface[index].y;
    }

    // Spike rejection, against the median of the samples that are actually touching. The median
    // rather than the mean because the thing being rejected is precisely an outlier, and an outlier
    // moves a mean towards itself.
    auto touching = std::vector<double>{};
    for (const auto depth : penetrations)
    {
        if (depth > 0.0)
        {
            touching.push_back(depth);
        }
    }

    if (touching.empty())
    {
        return patch;
    }

    std::sort(touching.begin(), touching.end());
    const auto median = touching[touching.size() / 2];
    const auto ceiling = median + sampling.spikeRejection;

    // --- the belt ---------------------------------------------------------------------------------
    //
    // Which elements the road is under, which pairs of them the belt bends over rather than shears
    // across, and then the bed itself. A rejected spike leaves no constraint behind: it is not road,
    // so nothing about it should hold an element up.
    auto constrained = std::vector<char>(hits.size(), char{0});
    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        const auto depth = penetrations[index];
        constrained[index] = hits[index].hit && depth > 0.0 && depth <= ceiling ? char{1} : char{0};
    }

    // Cut only where two samples both found road and disagree about which way it faces. A sample that
    // found nothing does not cut anything — the belt spanning a void is the case this model exists
    // for, and a missing normal is not evidence of a break.
    const auto breakBetween = [&](const std::size_t a, const std::size_t b)
    {
        if (!hits[a].hit || !hits[b].hit)
        {
            return char{0};
        }

        const auto cosine = std::clamp(glm::dot(hits[a].normal, hits[b].normal), -1.0, 1.0);

        return std::acos(cosine) > sampling.beltBreakAngle ? char{1} : char{0};
    };

    auto cutAcross = std::vector<char>(hits.size(), char{0});
    auto cutAlong = std::vector<char>(hits.size(), char{0});

    for (auto row = std::uint32_t{0}; row < sampling.along; row++)
    {
        for (auto column = std::uint32_t{0}; column + 1 < sampling.across; column++)
        {
            const auto left = static_cast<std::size_t>(row) * sampling.across + column;
            cutAcross[left] = breakBetween(left, left + 1);
        }
    }

    for (auto row = std::uint32_t{0}; row + 1 < sampling.along; row++)
    {
        for (auto column = std::uint32_t{0}; column < sampling.across; column++)
        {
            const auto near = static_cast<std::size_t>(row) * sampling.across + column;
            cutAlong[near] = breakBetween(near, near + sampling.across);
        }
    }

    // The element spacings are the grid's, so the shear rates fall out of the patch's own dimensions
    // and the belt's one stated length. They differ between the two axes because the grid does, which
    // is right: a grid too coarse in one direction to resolve the belt simply does not couple in it.
    const auto spacingAcross = sampling.across > 1 ? sampling.width / static_cast<double>(sampling.across - 1) : 0.0;
    const auto spacingAlong = sampling.along > 1 ? sampling.length / static_cast<double>(sampling.along - 1) : 0.0;

    const auto bed = solveBeltBed(penetrations, constrained, cutAcross, cutAlong, sampling.across, sampling.along,
                                  beltShearRate(spacingAcross, sampling.beltBridgingLength),
                                  beltShearRate(spacingAlong, sampling.beltBridgingLength), sampling.beltIterations);

    auto totalWeight = 0.0;
    auto weightedNormal = glm::dvec3(0.0);
    auto weightedCentre = glm::dvec3(0.0);
    auto weightedGrip = 0.0;
    auto weightedBumpiness = 0.0;
    // **The belt's deflection over the whole grid, and not the depths.** Those are the same sum on an
    // uncoupled bed, where an element deflects exactly as far as the road pushes it and no further;
    // they differ by precisely the work the coupling does, which is an element dragged inward by a
    // loaded neighbour carrying a share of the load rather than none. Summed over every element
    // including the ones touching nothing, because the rim feels every radial spring whether or not
    // the road is under it — and because the sum of the deflections is exactly the sum of the contact
    // pressures, which is the identity that says the coupling never invents load.
    auto summedPenetration = 0.0;
    for (const auto deflection : bed.deflection)
    {
        summedPenetration += deflection;
    }

    // Tracked across the load-carrying set only, and seeded outside the range of any depth that can
    // enter it — every accepted sample is strictly positive and bounded above by the ceiling, so the
    // first one replaces both ends.
    auto deepest = 0.0;
    auto shallowest = 0.0;

    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        const auto depth = penetrations[index];

        // Rejected outright rather than clamped to the ceiling, which was the first thing tried and
        // does not work: load share goes as depth, so a spike clamped to median + 50 mm still
        // outweighs a neighbour penetrating 10 mm by six to one and drags the plane over with it.
        // A sample this far past its neighbours is not road, and half-believing it is worse than
        // not believing it.
        //
        // **Load share is the element's own contact pressure and no longer its depth.** Those are the
        // same number, exactly, on an uncoupled bed — pressure is `k * depth` for a linear spring and
        // the constant cancels out of a weighted mean — and they part company where the belt does
        // work: an element the belt has lifted clear of the road presses on nothing however deep the
        // road under it looks, and one held down by its loaded neighbours presses harder than its own
        // depth would say. Weighting by depth would put the first of those into the patch centre.
        if (bed.pressure[index] <= 0.0)
        {
            continue;
        }

        const auto weight = bed.pressure[index];

        deepest = patch.contactingSamples == 0 ? depth : std::max(deepest, depth);
        shallowest = patch.contactingSamples == 0 ? depth : std::min(shallowest, depth);

        totalWeight += weight;
        patch.contactingSamples++;

        weightedNormal += weight * hits[index].normal;
        weightedCentre += weight * hits[index].point;

        const auto surface = hits[index].surface;
        const auto& material = surface < materials.size() ? materials[surface] : materials.front();
        weightedGrip += weight * material.gripMultiplier;
        weightedBumpiness += weight * material.bumpiness;
    }

    if (totalWeight <= 0.0)
    {
        return patch;
    }

    patch.inContact = true;
    patch.centre = weightedCentre / totalWeight;
    patch.normal = glm::normalize(weightedNormal);
    patch.gripMultiplier = weightedGrip / totalWeight;
    patch.bumpiness = weightedBumpiness / totalWeight;
    patch.penetration = summedPenetration / static_cast<double>(hits.size());
    patch.depthSpread = deepest - shallowest;

    return patch;
}

} // namespace raceengine
