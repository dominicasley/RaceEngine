module;

#include <expected>
#include <string>

export module raceengine.graphics:IRenderBackend;

import :IFrameCapture;
import :IFrameRecorder;
import :IGpuResourceFactory;

namespace raceengine
{

// One graphics device, wearing the three seams its consumers use. Only the composition root
// names this type: Engine owns the backend and hands every service the narrow interface it
// needs — IGpuResourceFactory to FboService, ShaderService and CubeMapService,
// IFrameRecorder to PresenterService and to Engine::step. A service that takes this type
// instead is a service that has been handed the whole device.
//
// What remains declared here is device lifecycle, which has no consumer but the composition
// root.
export class IRenderBackend : public IFrameRecorder, public IGpuResourceFactory, public IFrameCapture
{
public:
    // Brings the device up. Failure is a runtime condition — no driver, no suitable device,
    // no surface — so it is reported: what to do about an engine that cannot render is the
    // composition root's decision, not the backend's.
    [[nodiscard]] virtual std::expected<void, std::string> init() = 0;

    // The window's drawable size changed. It records the new extent and nothing else; the
    // frame belongs to Engine::step, and a resize callback runs between frames.
    virtual void setViewport(int width, int height) = 0;
};

} // namespace raceengine
