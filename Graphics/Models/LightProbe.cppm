module;

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include <glm/glm.hpp>

export module raceengine.graphics.models:LightProbe;

namespace raceengine
{

// Nine coefficients: the L2 band, which is where a cosine-convolved environment stops carrying
// signal. The convolution kernel's own coefficients fall off as 1/l^2 and the l = 3 band is
// identically zero, so an L2 projection reproduces diffuse irradiance to within about 1% of the
// exact integral for any environment. Storing more would store noise.
export inline constexpr unsigned int shCoefficientCount = 9;

// Irradiance as the shading side consumes it: already convolved with the clamped-cosine lobe and
// already divided by pi, so evaluating the basis in a direction gives outgoing diffuse *radiance*
// for a unit-albedo surface and the shader's only remaining job is to multiply by albedo.
//
// vec4 rather than vec3 because this array's destination is a std140 uniform block, where an array
// element is padded to 16 bytes regardless — carrying the padding here is what makes the C++
// object the GPU layout rather than something that has to be repacked on the way.
export using ShIrradiance = std::array<glm::vec4, shCoefficientCount>;

// Where a probe is in its capture cycle. A probe is not usable for shading until it has been
// through the whole of one: the faces it has drawn so far are a partial environment, and shading
// from those would light the world from whichever directions happened to be finished.
export enum class LightProbeState {
    // Nothing captured, or what was captured is stale. The scheduler picks these up.
    Dirty,
    // Some faces drawn, the rest still to come. Only the scheduler is in this state's business.
    Capturing,
    // Six faces drawn and prefiltered; waiting for the irradiance. The projection reads a buffer
    // the GPU filled, so it cannot run until the frame that filled it has completed — which is
    // why this is a state and not a step. The probe's *previous* irradiance keeps shading the
    // frame throughout, so a re-capture is a slide rather than a flash to black.
    Projecting,
    // Six faces drawn, prefiltered, and the irradiance projected. Shading reads this probe.
    Ready,
};

// One node of the image-based lighting graph: a point the world was photographed from, and the
// region of space that photograph is a good enough answer for.
//
// The volume is an axis-aligned box rather than a sphere because the things that bound indirect
// light are flat — a room stops at its walls, a street stops at its buildings — and a sphere
// covering a room's corners reaches through the walls to do it. `blendDistance` is the band
// inside each face over which this probe's weight ramps up, so two probes whose boxes overlap
// hand off continuously instead of switching at a surface the eye can see.
export struct LightProbe
{
    std::string name;
    glm::vec3 position{};
    // Half the box's side lengths, about `position`. Ignored when `global` is set.
    glm::vec3 halfExtents{1.0f};
    float blendDistance = 1.0f;
    // The scene's fallback probe: weight 1 everywhere, and the one every fragment falls back on
    // for whatever weight the local probes did not account for. A scene wants exactly one, placed
    // where it can see the sky — it is what makes "outside every local volume" a lit answer
    // rather than a black one.
    bool global = false;
    // How far the capture can see. The near plane also bounds the parallax proxy: a reflection is
    // re-projected onto the influence box, which is only a good approximation while the box is
    // roughly where the geometry is.
    float nearClippingPlane = 0.5f;
    float farClippingPlane = 5000.0f;

    LightProbeState state = LightProbeState::Dirty;
    // Which of the six faces the scheduler draws next. Capture is spread across frames — six
    // full scene passes in one frame is a visible hitch — so a probe carries its own progress.
    unsigned int captureFace = 0;
    // The slice of the backend's probe array this probe's prefiltered radiance lives in.
    // Assigned on first capture and held for the probe's life, which is what lets a re-capture
    // overwrite in place rather than shuffling every other probe's index.
    std::optional<unsigned int> arraySlice{};

    // Whether this probe's first capture has been reported. Per probe rather than per backend so
    // a scene says what every one of its probes recorded, once, and a re-capture stays silent.
    bool captureLogged = false;

    // Read by the frame's uniform upload every view; written by the backend once per capture.
    ShIrradiance irradiance{};

    // Marks the environment this probe recorded as no longer true — the sun moved, a door opened,
    // the time of day changed. The next scheduler pass re-draws all six faces. It does *not*
    // clear the irradiance: the stale answer keeps shading the frame until the new one lands,
    // which is what makes a re-capture invisible rather than a flicker to black.
    void invalidate()
    {
        state = LightProbeState::Dirty;
        captureFace = 0;
    }
};

} // namespace raceengine
