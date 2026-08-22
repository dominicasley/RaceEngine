module;

#include <algorithm>
#include <expected>
#include <string>

#include <glm/glm.hpp>
#include <spdlog/logger.h>

export module raceengine.graphics:BloomService;

import :CameraService;
import :FboService;
import :PostProcessing;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class BloomService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    PostProcessService& postProcessService;
    CameraService& cameraService;

public:
    explicit BloomService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, FboService& fboService,
                          PostProcessService& postProcessService, CameraService& cameraService);
    // Builds the two chains and every pass over them, and appends the passes to the camera's own
    // list in the order they have to run. Enable it *before* the pass that tone maps: the spill is
    // added to the frame in the exposed domain, so what reads it has to come after what writes it.
    [[nodiscard]] std::expected<void, std::string> enable(Camera& camera, const CreateBloomDTO& dto) const;
    // Both chains at the window's new size. Half of it, as they were built.
    [[nodiscard]] std::expected<void, std::string> resize(const Camera& camera, int width, int height) const;
};

} // namespace raceengine
