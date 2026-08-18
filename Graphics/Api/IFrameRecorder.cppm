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
    [[nodiscard]] virtual bool beginFrame() = 0;

    // One view: the camera's scene pass into that camera's own render target, followed by
    // that camera's post-process chain. Each pass establishes the state it reads, so views
    // are independent of each other and of their order.
    virtual void recordView(Scene& scene, Camera& camera, float delta) = 0;

    // The frame's last pass: `attachment` sampled through `shader` onto the backbuffer.
    // Recording only — endFrame is what puts it on screen.
    virtual void recordPresent(const Resource<Shader>& shader, const Resource<FboAttachment>& attachment) = 0;

    // Submits everything recorded since beginFrame and presents it, closing the frame.
    virtual void endFrame() = 0;
};

} // namespace raceengine
