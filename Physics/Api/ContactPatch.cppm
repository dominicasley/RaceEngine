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
    std::uint32_t across = 3;
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
    // sample dominates exactly when it is least trustworthy. Fifty millimetres is well past any
    // real kerb this grid would straddle and well under a mesh fault.
    double spikeRejection = 0.05;
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

    // The *mean* over the whole grid, including the samples touching nothing — which is a different
    // aggregate from the three above and deliberately so. Vertical force goes as the total
    // compression across the patch, so a wheel with half its patch in the air must carry about half
    // the load. Weighting this one by load share too would report the deep half's penetration and
    // hand back a wheel that is carrying full load on half a contact patch.
    //
    // It is a **quadrature, and it converges rather than being exact**: even on perfectly flat
    // ground the compression varies across the patch because the tread curves away from it, so a
    // 3x3 grid and a 7x7 grid over the same wheel disagree by something like 15%. That is not an
    // error to be fixed — it is the accuracy of three points across a curve — but it does mean the
    // sample count is part of a vehicle's configuration rather than free to change under a car that
    // has already been given a spring rate. The normal, the grip and the patch centre have no such
    // dependence.
    double penetration = 0.0;

    std::uint32_t contactingSamples = 0;
    std::uint32_t totalSamples = 0;
};

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

    auto totalWeight = 0.0;
    auto weightedNormal = glm::dvec3(0.0);
    auto weightedCentre = glm::dvec3(0.0);
    auto weightedGrip = 0.0;
    auto weightedBumpiness = 0.0;
    auto summedPenetration = 0.0;

    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        const auto depth = penetrations[index];

        // Rejected outright rather than clamped to the ceiling, which was the first thing tried and
        // does not work: load share goes as depth, so a spike clamped to median + 50 mm still
        // outweighs a neighbour penetrating 10 mm by six to one and drags the plane over with it.
        // A sample this far past its neighbours is not road, and half-believing it is worse than
        // not believing it.
        if (depth <= 0.0 || depth > ceiling)
        {
            continue;
        }

        // Load share goes as compression, which for a linear tire spring is what it is.
        const auto weight = depth;

        totalWeight += weight;
        summedPenetration += depth;
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

    return patch;
}

} // namespace raceengine
