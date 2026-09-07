// LightProbeService bodies. Declarations are in Graphics/Services/LightProbeService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

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

std::vector<ShIrradiance> LightProbeService::irradianceSnapshot(const Scene& scene) const
{
    auto snapshot = std::vector<ShIrradiance>();
    snapshot.reserve(scene.probes.size());

    for (const auto& probe : scene.probes)
    {
        snapshot.push_back(probe.irradiance);
    }

    return snapshot;
}

bool LightProbeService::restoreIrradiance(Scene& scene, const std::span<const ShIrradiance> snapshot) const
{
    if (snapshot.size() != scene.probes.size())
    {
        return false;
    }

    auto index = size_t{0};

    for (auto& probe : scene.probes)
    {
        probe.irradiance = snapshot[index];
        probe.irradianceReady = true;

        // The pool's worth at the front still capture: they are the probes a frame can reflect in,
        // and a snapshot carries no reflection. Everything past them is finished here and now.
        if (index >= maxIblProbes)
        {
            probe.state = LightProbeState::Ready;
            probe.captureLogged = true;
        }

        index++;
    }

    return true;
}

bool LightProbeService::everyProbeCaptured(const Scene& scene) const
{
    return std::ranges::all_of(scene.probes, [](const LightProbe& probe) { return probe.irradianceReady; });
}

} // namespace raceengine
