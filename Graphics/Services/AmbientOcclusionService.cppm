module;

#include <algorithm>
#include <expected>
#include <string>

#include <glm/glm.hpp>
#include <spdlog/logger.h>

export module raceengine.graphics:AmbientOcclusionService;

import :FboService;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class AmbientOcclusionService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    PostProcessService& postProcessService;

public:
    explicit AmbientOcclusionService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                                     FboService& fboService, PostProcessService& postProcessService);
    // Builds the three buffers this camera's occlusion is gathered through and turns it on. They are
    // the view's own size: occlusion is sampled by the shading pass at the pixel it is shading, so a
    // buffer at another resolution would be an upsample in the one place a wrong answer looks like
    // geometry.
    [[nodiscard]] std::expected<void, std::string> enable(Camera& camera, const CreateAmbientOcclusionDTO& dto) const;
    // Every buffer this camera gathers through, at the window's new size. All three follow the view
    // because the shading pass reads the result at its own pixel: one of them left behind puts a
    // scaled ghost of the scene onto every surface, which looks like the occlusion being wrong
    // rather than like the buffer being the wrong size. A camera that gathers nothing is not a
    // failure — there is simply nothing to rebuild.
    [[nodiscard]] std::expected<void, std::string> resize(const Camera& camera, int width, int height) const;
};

} // namespace raceengine
