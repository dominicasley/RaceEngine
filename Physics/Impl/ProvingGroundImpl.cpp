// ProvingGround bodies. Declarations are in Api/ProvingGround.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <string>
#include <vector>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] std::vector<SurfaceMaterial> defaultSurfaceMaterials()
{
    return {SurfaceMaterial{.gripMultiplier = 1.00, .bumpiness = 0.0015, .kind = SurfaceKind::Tarmac},
            SurfaceMaterial{.gripMultiplier = 0.85, .bumpiness = 0.0060, .kind = SurfaceKind::Kerb},
            SurfaceMaterial{.gripMultiplier = 0.42, .bumpiness = 0.0200, .kind = SurfaceKind::Grass}};
}

[[nodiscard]] ProvingGroundDescriptor defaultProvingGround()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 60.0, .to = 90.0},
                           Feature{.kind = FeatureKind::SurfaceBoundary, .from = 100.0, .to = 130.0},
                           Feature{.kind = FeatureKind::Camber, .from = 140.0, .to = 170.0},
                           Feature{.kind = FeatureKind::Ramp, .from = 185.0, .to = 200.0}};

    return descriptor;
}

[[nodiscard]] SurfaceSample sampleProvingGround(const ProvingGroundDescriptor& descriptor, const double x,
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

[[nodiscard]] std::expected<SurfaceMesh, std::string> generateProvingGround(const ProvingGroundDescriptor& descriptor)
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
