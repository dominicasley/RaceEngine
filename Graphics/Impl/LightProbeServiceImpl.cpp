// LightProbeService bodies. Declarations are in Graphics/Services/LightProbeService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cstddef>
#include <string>

#include <glm/glm.hpp>

module raceengine.graphics;

import :LightProbeService;
import :RenderContract;
import raceengine.graphics.models;

namespace raceengine
{

LightProbe& LightProbeService::createProbe(Scene& scene, const CreateLightProbeDTO& descriptor) const
{
    // A blend band of nothing is a discontinuity at the box face, which reads as a seam ruled
    // across the ground exactly like an unblended cascade split does. A tenth of the smallest
    // half-extent is a band wide enough to hide the handover and narrow enough that the probe
    // still fills its own volume.
    auto blendDistance = descriptor.blendDistance;
    if (blendDistance <= 0.0f)
    {
        blendDistance =
            glm::min(glm::min(descriptor.halfExtents.x, descriptor.halfExtents.y), descriptor.halfExtents.z) * 0.1f;
    }

    return scene.probes.emplace_back(LightProbe{.name = descriptor.name,
                                                .position = descriptor.position,
                                                .halfExtents = descriptor.halfExtents,
                                                .blendDistance = blendDistance,
                                                .global = descriptor.global,
                                                .nearClippingPlane = descriptor.nearClippingPlane,
                                                .farClippingPlane = descriptor.farClippingPlane});
}

void LightProbeService::invalidateAll(Scene& scene) const
{
    for (auto& probe : scene.probes)
    {
        probe.invalidate();
    }
}

unsigned int LightProbeService::activeProbeCount(const Scene& scene) const
{
    return static_cast<unsigned int>(std::min(scene.probes.size(), static_cast<size_t>(maxIblProbes)));
}

} // namespace raceengine
