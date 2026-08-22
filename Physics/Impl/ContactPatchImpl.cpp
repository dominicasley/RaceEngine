// ContactPatch bodies. Declarations are in Api/ContactPatch.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] double beltShearRate(const double spacing, const double bridgingLength)
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

[[nodiscard]] BeltBed solveBeltBed(const std::vector<double>& gaps, const std::vector<char>& constrained,
                                   const std::vector<char>& cutAcross, const std::vector<char>& cutAlong,
                                   const std::uint32_t across, const std::uint32_t along, const double shearAcross,
                                   const double shearAlong, const std::uint32_t iterations)
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

[[nodiscard]] ContactSampleGeometry contactPatchSamples(const WheelPose& wheel, const ContactPatchSampling& sampling)
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

[[nodiscard]] ContactPatch aggregateContactPatch(const ContactSampleGeometry& geometry,
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
