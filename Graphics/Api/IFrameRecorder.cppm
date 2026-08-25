module;

// GMF-included because this interface *names* glm::vec4: reachability through another module's
// interface is not visibility, which is the rule that makes a module unit state its own includes.
#include <glm/glm.hpp>

export module raceengine.graphics:IFrameRecorder;

import raceengine.graphics.models;

namespace raceengine
{

// The frame, in the order the composition root records it. Engine::step owns the whole
// sequence — beginFrame, one recordView per camera, one recordPresent, endFrame — and no
// method here departs from it: nothing opens a frame implicitly, and endFrame is the only
// one that submits or presents. That is what lets N cameras render N views and reach the
// screen in a single present, identically on either backend.
//
// Calls outside that order are a caller bug, not a runtime condition: recordView,
// recordPresent and endFrame all require the frame beginFrame opened, and beginFrame
// returning false means this step has no frame at all.
export class IFrameRecorder
{
public:
    IFrameRecorder() = default;
    IFrameRecorder(const IFrameRecorder&) = delete;
    IFrameRecorder(IFrameRecorder&&) = delete;
    IFrameRecorder& operator=(const IFrameRecorder&) = delete;
    IFrameRecorder& operator=(IFrameRecorder&&) = delete;
    virtual ~IFrameRecorder() = default;

    // Opens the frame. False means there is nothing to render into this step — a swapchain
    // that went out of date, or a minimised window — so the caller records nothing and does
    // not call endFrame.
    //
    // `simulationTime` is the simulated instant this frame renders, in seconds — the engine's tick
    // count times its fixed step, and the only clock anything temporal in a shader may read. Wall
    // time would put a different image on disk every capture run; this is a function of the frame
    // number, so frame N is the same instant on every machine.
    [[nodiscard]] virtual bool beginFrame(double simulationTime) = 0;

    // One view: the camera's scene pass into that camera's own render target, followed by
    // that camera's post-process chain. Each pass establishes the state it reads, so views
    // are independent of each other and of their order.
    virtual void recordView(Scene& scene, Camera& camera, float delta) = 0;

    // One step of a light probe's capture: draws the next face of `probe`'s environment, and,
    // once the sixth has been drawn, prefilters the result and projects its irradiance. The
    // probe's own state is what says which step this is, and the backend advances it.
    //
    // A step rather than a whole capture because six full scene passes in one frame is a hitch
    // the eye sees, and because a probe is captured far more often than once: a time-of-day
    // change invalidates every probe in the scene.
    //
    // Recorded before any Scene camera and after the cascades, for the reason CameraRole gives:
    // a probe shades from the cascades, and the Scene cameras shade from the probe.
    virtual void recordProbeCapture(Scene& scene, LightProbe& probe) = 0;

    // The occlusion this camera is about to shade with: draws its view once more into the prepass
    // buffer — view-space normals and depth, no shading, no sky — then runs the gather and the blur
    // that turn that buffer into one visibility term per pixel.
    //
    // Recorded immediately *before* that camera's own recordView, which is the producer/consumer
    // rule CameraRole states applied one more time: the shading pass samples the buffer this fills,
    // and a consumer recorded first would read it before it was written. A camera with no occlusion
    // enabled records nothing here and shades against the 1x1 white image the backend binds instead.
    virtual void recordAmbientOcclusion(Scene& scene, Camera& camera, float delta) = 0;

    // One step of a camera's exposure meter: copies the single texel its reduction chain was
    // reduced into back towards the CPU, and collects the copy the last call queued once the
    // submission carrying it has completed. The camera's own AutoExposure is where the collected
    // reading lands, and it holds the previous one until then.
    //
    // Recorded immediately after that camera's own recordView, because the chain it copies from is
    // the tail of that view's post-process passes and the copy has to be a command in the same
    // frame. Deferred rather than waited on for the reason the light probe projection is: a fence
    // taken in the middle of a frame is a stall, and the number is a frame or two stale either way.
    virtual void recordAutoExposure(Camera& camera) = 0;

    // The frame's last pass: the presenter's attachment sampled through its shader onto the
    // backbuffer, with its own numbers and its colour grade. The whole Presenter rather than its
    // parts because they travel together and a fourth loose argument is a fourth thing to keep in
    // the same order at both ends. Recording only — endFrame is what puts it on screen.
    virtual void recordPresent(const Presenter& presenter) = 0;

    // Submits everything recorded since beginFrame and presents it, closing the frame.
    virtual void endFrame() = 0;
};

} // namespace raceengine
