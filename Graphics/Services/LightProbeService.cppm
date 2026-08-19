module;

#include <algorithm>
#include <cstddef>
#include <string>

#include <glm/glm.hpp>

export module raceengine.graphics:LightProbeService;

import :RenderContract;
import raceengine.graphics.models;

namespace raceengine
{

// What a game states when it places a probe. A box and a blend band, because that is the whole of
// the authoring: everything else about a probe — its resolution, its mip chain, which array slice
// it lands in, when it is re-drawn — is the engine's.
export struct CreateLightProbeDTO
{
    std::string name;
    glm::vec3 position{};
    // Half the box's side lengths. Ignored for a global probe, which has no bound.
    glm::vec3 halfExtents{1.0f};
    // How far inside each face this probe's weight ramps from nothing to full. Zero is legal and
    // means a hard edge, which is visible; the default is a tenth of the smallest half-extent,
    // applied by createProbe when this is left at zero.
    float blendDistance = 0.0f;
    bool global = false;
    float nearClippingPlane = 0.5f;
    float farClippingPlane = 5000.0f;
};

// The image-based lighting graph, from the game's side: place probes, and say when what they
// recorded has stopped being true.
//
// It owns no device and records nothing. A probe is scene data — the backend reads the scene's
// probes the same way it reads its lights — so this service only writes that data and applies the
// defaults a game should not have to state.
export class LightProbeService
{
public:
    // Appends a probe to the scene. It starts Dirty, so the frame's scheduler picks it up and
    // captures it over the following frames; nothing has to ask for the first capture.
    //
    // A scene past maxIblProbes still takes the probe — the container has no limit — but the
    // shading side reads the first maxIblProbes of them, so the extras light nothing. The count
    // is reported rather than clamped: dropping a probe a level asked for silently is how a level
    // ends up with lighting nobody can account for.
    LightProbe& createProbe(Scene& scene, const CreateLightProbeDTO& descriptor) const;

    // The time of day changed, a door opened, the sun moved: every probe's recorded environment is
    // stale. They re-capture over the frames that follow, holding their previous irradiance until
    // the new one lands, so the transition is a slide rather than a flash.
    void invalidateAll(Scene& scene) const;

    // How many of this scene's probes the frame will actually shade from.
    [[nodiscard]] unsigned int activeProbeCount(const Scene& scene) const;
};

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
