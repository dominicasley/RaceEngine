module;

#include <cmath>
#include <expected>
#include <string>

#include <glm/glm.hpp>
#include <spdlog/logger.h>

export module raceengine.graphics:AutoExposureService;

import :CameraService;
import :FboService;
import :PhysicalCamera;
import :PostProcessing;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class AutoExposureService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    PostProcessService& postProcessService;
    CameraService& cameraService;

public:
    explicit AutoExposureService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                                 FboService& fboService, PostProcessService& postProcessService,
                                 CameraService& cameraService);
    // Builds the reduction this camera meters from and turns metering on. One buffer with a full
    // mip chain and one pass per level of it, appended to the camera's own post-process list: the
    // first reads the frame the camera drew and writes its log luminance, and every level below
    // averages the level above out of the buffer it is writing into.
    [[nodiscard]] std::expected<void, std::string> enable(Camera& camera, const CreateAutoExposureDTO& dto) const;
    // One frame of adaptation, advanced by the fixed steps the frame simulated rather than by how
    // long the frame took. That is the whole of what keeps the capture gate exact: `ticks` is a
    // function of the frame number under RACEENGINE_DUMP_FRAME, and elapsed time is not.
    void update(Camera& camera, unsigned int ticks, float tickSeconds) const;
};

} // namespace raceengine
