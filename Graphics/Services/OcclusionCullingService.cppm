module;

#include <expected>
#include <string>

#include <spdlog/logger.h>

export module raceengine.graphics:OcclusionCullingService;

import :FboService;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class OcclusionCullingService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    PostProcessService& postProcessService;

public:
    explicit OcclusionCullingService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                                     FboService& fboService, PostProcessService& postProcessService);
    // Builds the one buffer the culler tests against and turns it on.
    //
    // **Ambient occlusion has to be enabled on this camera first**, and this reports rather than
    // assumes it: the reduction's input is the prepass buffer's distance channel, so without a
    // prepass there is nothing to reduce and no occluders to have. That ordering is the only thing
    // a caller has to get right.
    //
    // There is no resize. The grid is a fixed number of cells addressed through clip space, so a
    // window that changes size changes what a cell covers and nothing else — which is what lets the
    // readback buffer and the pending copy be sized once, at startup, and never be a frame behind
    // the thing they are copying.
    [[nodiscard]] std::expected<void, std::string> enable(Camera& camera,
                                                          const CreateOcclusionCullingDTO& dto) const;
};

} // namespace raceengine
