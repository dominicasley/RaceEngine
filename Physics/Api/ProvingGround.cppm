module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:ProvingGround;

namespace raceengine
{

// The surface the vehicle is validated against, generated rather than authored.
//
// A flat plane would let every one of the contact-patch failures pass unnoticed — aggregation
// across a tilted patch, a migrating patch centre, blended grip at a boundary — because none of
// them has anything to act on. So the ground carries a kerb, a surface boundary, a camber change
// and a ramp from the first day, and it is *generated* so that it is deterministic, diffable, and
// does not depend on an asset that can be lost. (One already was: see the note on plane.glb.)
//
// SI, like everything else under Physics: metres. +x is lateral, +z is along the ground, +y is up,
// matching the engine's handedness.

export enum class SurfaceKind : std::uint32_t { Tarmac, Kerb, Grass };

// What a surface is worth to a tire, separately from where it is. Kept beside the mesh rather than
// baked into it because the tire model scales its own mu by this at runtime — the seam the deferred
// thermal model needs, and the reason `gripMultiplier` is not folded into a Pacejka coefficient.
export struct SurfaceMaterial
{
    double gripMultiplier = 1.0;
    // Amplitude, in metres, of the fine surface roughness a tire feels but the mesh does not
    // resolve. Nothing consumes it this milestone; it is a channel the sampler already carries.
    double bumpiness = 0.0;
    SurfaceKind kind = SurfaceKind::Tarmac;
};

export enum class FeatureKind : std::uint32_t {
    // A raised kerb parallel to the direction of travel, so a wheel can be run onto it at an angle
    // and half-mount it. This is the case single-ray contact gets visibly wrong.
    Kerb,
    // Tarmac one side, grass the other, with the small lip a real one has. A wheel straddling it
    // must come out with grip between the two rather than snapping from one to the other.
    SurfaceBoundary,
    // The ground banks, so the contact plane is not the world's horizontal and the suspension has
    // to be solved against a surface that is not flat under it.
    Camber,
    // Up and then nothing: the launch that criterion 11 asks for a clean arc off.
    Ramp,
    // A constant grade along the direction of travel. What criteria 2 and 8 need: somewhere to park
    // a car on the brakes and find out whether it stays there, and somewhere to creep up and down
    // at walking pace and find out whether the tire model is still finite when it does.
    Slope,
    // A wall down one side. The only feature that is not a height — a heightfield cannot express a
    // vertical face at all — so it is emitted as its own geometry after the grid, and it is the one
    // thing criterion 2's first half needs: something for a car to rest against.
    Barrier
};

export struct Feature
{
    FeatureKind kind = FeatureKind::Kerb;
    // The band this feature occupies along z, in metres. Bands are evaluated in order and a later
    // one wins, so overlapping them is a way of composing rather than an error.
    double from = 0.0;
    double to = 0.0;
};

export struct ProvingGroundDescriptor
{
    double length = 200.0;
    double width = 40.0;
    // The grid pitch. A tire's contact patch is about 0.2 m across, and the sample grid inside it
    // is finer still, so anything much coarser than this would have the patch spanning one triangle
    // and aggregating nothing.
    double cellSize = 0.25;

    // Height of the kerb's flat top, and the width of the chamfer up onto it.
    double kerbHeight = 0.05;
    double kerbChamfer = 0.30;
    // Where the kerb's inner edge sits, measured from the centreline towards +x.
    double kerbInnerEdge = 3.0;

    // How far the grass sits below the tarmac at the boundary.
    double boundaryLip = 0.02;
    // Peak bank angle of the camber section, radians. Six degrees.
    double camberAngle = 0.10472;
    double rampHeight = 1.20;
    // Fifteen degrees, which is what criterion 2 asks a parked car to hold on.
    double slopeAngle = 0.26180;

    // Where the barrier stands and how tall it is. Positive x, so a car pushed to its right finds it.
    double barrierX = 4.0;
    double barrierHeight = 0.9;

    // How far a feature takes to come up to full height at each end of its band, in metres.
    //
    // Without it a feature switches on abruptly at its own boundary, and a kerb that begins as a
    // fifty-millimetre vertical wall across the road is not a kerb — it is a step, and a wheel
    // hitting it produces a load spike that looks exactly like a contact bug. Real kerbs are ramped
    // at the ends and so is this one.
    double featureEasing = 1.0;

    std::vector<Feature> features;
};

export struct SurfaceSample
{
    double height = 0.0;
    SurfaceKind kind = SurfaceKind::Tarmac;
};

export struct SurfaceMesh
{
    std::vector<glm::dvec3> vertices;
    // Three per triangle, counter-clockwise seen from above, which is the winding the engine's
    // front face already is.
    std::vector<std::uint32_t> indices;
    // One per triangle, indexing `materials`. Per-triangle rather than per-vertex because a
    // boundary between two surfaces is a line the mesh is cut along, not a blend across a face —
    // the blending belongs to the sampler that aggregates a contact patch, where it can be weighted
    // by how much of the patch is on each side.
    std::vector<std::uint32_t> surfaces;
    std::vector<SurfaceMaterial> materials;

    [[nodiscard]] std::size_t triangleCount() const
    {
        return indices.size() / 3;
    }
};

// The materials the generator emits, in `SurfaceKind` order so a kind indexes its own material.
export [[nodiscard]] std::vector<SurfaceMaterial> defaultSurfaceMaterials()
{
    return {SurfaceMaterial{.gripMultiplier = 1.00, .bumpiness = 0.0015, .kind = SurfaceKind::Tarmac},
            SurfaceMaterial{.gripMultiplier = 0.85, .bumpiness = 0.0060, .kind = SurfaceKind::Kerb},
            SurfaceMaterial{.gripMultiplier = 0.42, .bumpiness = 0.0200, .kind = SurfaceKind::Grass}};
}

// The layout the acceptance criteria need: a long flat run to settle, accelerate, brake and hold a
// skidpad on, then each feature in its own band with flat ground either side so a test can drive
// onto exactly one of them.
export [[nodiscard]] ProvingGroundDescriptor defaultProvingGround()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 60.0, .to = 90.0},
                           Feature{.kind = FeatureKind::SurfaceBoundary, .from = 100.0, .to = 130.0},
                           Feature{.kind = FeatureKind::Camber, .from = 140.0, .to = 170.0},
                           Feature{.kind = FeatureKind::Ramp, .from = 185.0, .to = 200.0}};

    return descriptor;
}

// Height and surface at a point, which is the whole of the ground's definition: the mesh below is
// only this function sampled on a grid. Pure and total — every (x, z) has an answer, including
// outside the ground's extent, so a query that walks off the edge reads flat tarmac rather than
// nothing.
export [[nodiscard]] SurfaceSample sampleProvingGround(const ProvingGroundDescriptor& descriptor, const double x,
                                                       const double z)
{
    auto sample = SurfaceSample{};

    for (const auto& feature : descriptor.features)
    {
        if (z < feature.from || z >= feature.to)
        {
            continue;
        }

        // How far into its own band this feature has come up, 0 at each end and 1 in the middle.
        // Applied to the features that would otherwise begin as a step; the camber has its own
        // easing across the whole band and the ramp starts at nothing by construction.
        const auto easing = std::max(descriptor.featureEasing, 1e-9);
        const auto entering = std::clamp((z - feature.from) / easing, 0.0, 1.0);
        const auto leaving = std::clamp((feature.to - z) / easing, 0.0, 1.0);
        const auto blend = std::min(entering, leaving);

        switch (feature.kind)
        {
        case FeatureKind::Kerb:
        {
            // Flat top past the chamfer, a linear ramp across it, and nothing inboard of the edge.
            const auto across = x - descriptor.kerbInnerEdge;
            if (across > 0.0)
            {
                const auto rise = across < descriptor.kerbChamfer ? across / descriptor.kerbChamfer : 1.0;
                sample.height = descriptor.kerbHeight * rise * blend;
                sample.kind = SurfaceKind::Kerb;
            }
            break;
        }
        case FeatureKind::SurfaceBoundary:
        {
            if (x > 0.0)
            {
                sample.height = -descriptor.boundaryLip * blend;
                sample.kind = SurfaceKind::Grass;
            }
            break;
        }
        case FeatureKind::Camber:
        {
            // Eased in and out over the band with a raised cosine, so the entry is not a crease the
            // suspension has to absorb as a step.
            const auto through = (z - feature.from) / (feature.to - feature.from);
            const auto eased = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * through);
            sample.height = x * std::tan(descriptor.camberAngle * eased);
            break;
        }
        case FeatureKind::Ramp:
        {
            const auto through = (z - feature.from) / (feature.to - feature.from);
            sample.height = descriptor.rampHeight * through;
            break;
        }
        case FeatureKind::Barrier:
        {
            // Contributes no height: a wall is not a value this function can return, and the mesh
            // generator emits it separately.
            break;
        }
        case FeatureKind::Slope:
        {
            // No easing: a grade that faded in at its ends would not be a constant grade, which is
            // the one thing this feature is for.
            sample.height = (z - feature.from) * std::tan(descriptor.slopeAngle);
            break;
        }
        }
    }

    return sample;
}

// The generated mesh. Reported rather than returned bare: a descriptor with no extent, or a cell
// size that would make one, is a configuration error and not something to hand back an empty mesh
// for.
export [[nodiscard]] std::expected<SurfaceMesh, std::string>
generateProvingGround(const ProvingGroundDescriptor& descriptor)
{
    if (!(descriptor.cellSize > 0.0))
    {
        return std::unexpected("the proving ground needs a positive cell size");
    }

    if (!(descriptor.length > 0.0) || !(descriptor.width > 0.0))
    {
        return std::unexpected("the proving ground needs a positive length and width");
    }

    const auto cellsAcross = static_cast<std::size_t>(std::llround(descriptor.width / descriptor.cellSize));
    const auto cellsAlong = static_cast<std::size_t>(std::llround(descriptor.length / descriptor.cellSize));

    if (cellsAcross < 1 || cellsAlong < 1)
    {
        return std::unexpected("the proving ground is smaller than one cell");
    }

    auto mesh = SurfaceMesh{};
    mesh.materials = defaultSurfaceMaterials();
    mesh.vertices.reserve((cellsAcross + 1) * (cellsAlong + 1));
    mesh.indices.reserve(cellsAcross * cellsAlong * 6);
    mesh.surfaces.reserve(cellsAcross * cellsAlong * 2);

    const auto positionOf = [](const std::size_t index, const std::size_t cells, const double extent)
    {
        return -0.5 * extent + static_cast<double>(index) * extent / static_cast<double>(cells);
    };

    for (auto alongIndex = std::size_t{0}; alongIndex <= cellsAlong; alongIndex++)
    {
        // z runs from zero rather than from the centre: the features are laid out in distance
        // travelled, and a test that drives forward from the origin should meet them in order.
        const auto z = static_cast<double>(alongIndex) * descriptor.length / static_cast<double>(cellsAlong);

        for (auto acrossIndex = std::size_t{0}; acrossIndex <= cellsAcross; acrossIndex++)
        {
            const auto x = positionOf(acrossIndex, cellsAcross, descriptor.width);
            mesh.vertices.emplace_back(x, sampleProvingGround(descriptor, x, z).height, z);
        }
    }

    const auto rowStride = cellsAcross + 1;
    for (auto alongIndex = std::size_t{0}; alongIndex < cellsAlong; alongIndex++)
    {
        for (auto acrossIndex = std::size_t{0}; acrossIndex < cellsAcross; acrossIndex++)
        {
            const auto corner = alongIndex * rowStride + acrossIndex;
            const auto a = static_cast<std::uint32_t>(corner);
            const auto b = static_cast<std::uint32_t>(corner + 1);
            const auto c = static_cast<std::uint32_t>(corner + rowStride);
            const auto d = static_cast<std::uint32_t>(corner + rowStride + 1);

            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});

            // The cell's material is read at its centre, so a triangle belongs wholly to one
            // surface. A cell straddling a boundary resolves to whichever side its centre is on,
            // which is what makes the boundary a line in the mesh at the grid's pitch rather than
            // a gradient across it.
            const auto centreX = 0.5 * (mesh.vertices[a].x + mesh.vertices[d].x);
            const auto centreZ = 0.5 * (mesh.vertices[a].z + mesh.vertices[d].z);
            const auto kind = static_cast<std::uint32_t>(sampleProvingGround(descriptor, centreX, centreZ).kind);

            mesh.surfaces.push_back(kind);
            mesh.surfaces.push_back(kind);
        }
    }

    // Barriers, after the grid. Two triangles per span of the band, standing on the ground the grid
    // just laid down, facing inboard so a car on the road side meets the front of them.
    for (const auto& feature : descriptor.features)
    {
        if (feature.kind != FeatureKind::Barrier)
        {
            continue;
        }

        const auto spans = std::max(
            std::size_t{1},
            static_cast<std::size_t>(std::llround((feature.to - feature.from) / std::max(descriptor.cellSize, 1e-6))));

        for (auto span = std::size_t{0}; span <= spans; span++)
        {
            const auto z =
                feature.from + (feature.to - feature.from) * static_cast<double>(span) / static_cast<double>(spans);
            const auto ground = sampleProvingGround(descriptor, descriptor.barrierX, z).height;

            mesh.vertices.emplace_back(descriptor.barrierX, ground, z);
            mesh.vertices.emplace_back(descriptor.barrierX, ground + descriptor.barrierHeight, z);
        }

        const auto base = static_cast<std::uint32_t>(mesh.vertices.size() - (spans + 1) * 2);
        for (auto span = std::size_t{0}; span < spans; span++)
        {
            const auto low = base + static_cast<std::uint32_t>(span * 2);
            const auto high = low + 1;
            const auto nextLow = low + 2;
            const auto nextHigh = low + 3;

            // Wound so the face points towards -x, which is the side the road is on — and that is
            // worth checking rather than assuming. Wound the other way the wall is still there and
            // still solid to a ray cast, because a ray does not care which side of a triangle it
            // meets; but a *shape* query ignores back faces, so a car drives straight through a wall
            // that ray casts happily report. The winding test below is what pins it.
            mesh.indices.insert(mesh.indices.end(), {low, nextLow, high, high, nextLow, nextHigh});
            mesh.surfaces.push_back(static_cast<std::uint32_t>(SurfaceKind::Tarmac));
            mesh.surfaces.push_back(static_cast<std::uint32_t>(SurfaceKind::Tarmac));
        }
    }

    return mesh;
}

} // namespace raceengine
