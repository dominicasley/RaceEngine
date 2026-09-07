module;

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

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

    // The diffuse half of every probe in the scene, in the scene's own order.
    //
    // This is the half worth keeping between runs. It is nine coefficients — 144 bytes a probe —
    // and it is good from anywhere, where the specular half is a megabyte of prefiltered cube that
    // only exists for the handful of probes the frame can reflect in. A game that places hundreds
    // of probes spends most of its startup photographing them; taking this once and handing it
    // back on the next run is what turns that into a first-run cost.
    [[nodiscard]] std::vector<ShIrradiance> irradianceSnapshot(const Scene& scene) const;

    // Hands a snapshot back to the scene's probes, and answers whether it took.
    //
    // **Refused unless the snapshot is exactly as long as the probe list.** A snapshot taken
    // against a different layout would light every street with a different street's bounce, and
    // that is the kind of wrong nobody traces back to a cache file. It is the caller's job to
    // refuse it for the other reason too — a snapshot taken under a different sun is the same
    // fault and this cannot see the sun.
    //
    // The first `maxIblProbes` probes are handed their irradiance and left **Dirty** anyway: they
    // are the ones that will hold a specular slice, a snapshot carries no specular, and so they
    // have to be photographed whatever this file says. They shade from the restored irradiance
    // while they wait, so nothing starts black. Every probe past them is restored outright and is
    // never captured at all, which is the whole of the saving.
    [[nodiscard]] bool restoreIrradiance(Scene& scene, std::span<const ShIrradiance> snapshot) const;

    // Whether every probe in the scene now holds a photograph. This is what says a snapshot is
    // worth writing out: taken early it would record the probes that had not been reached yet as
    // black, and black is a value a cache cannot tell from a dark street.
    [[nodiscard]] bool everyProbeCaptured(const Scene& scene) const;
};

} // namespace raceengine
