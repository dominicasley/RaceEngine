// OcclusionCullingService bodies. Declarations are in Graphics/Services/OcclusionCullingService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <string>

#include <spdlog/logger.h>

module raceengine.graphics;

import :FboService;
import :OcclusionCullingService;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

OcclusionCullingService::OcclusionCullingService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                                                 FboService& fboService, PostProcessService& postProcessService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    postProcessService(postProcessService)
{
}

std::expected<void, std::string> OcclusionCullingService::enable(Camera& camera,
                                                                 const CreateOcclusionCullingDTO& dto) const
{
    const auto& occlusion = camera.ambientOcclusion;

    if (!occlusion.enabled || !occlusion.prepass.has_value())
    {
        return std::unexpected("occlusion culling reduces the ambient occlusion prepass, and this camera has none: "
                               "enable ambient occlusion on it first");
    }

    const auto* prepassBuffer = memoryStorageService.frameBuffers.find(occlusion.prepass.value());
    if (prepassBuffer == nullptr)
    {
        return std::unexpected("the ambient occlusion prepass handle names no live framebuffer");
    }

    // Resolved before the grid is created rather than after: this is a borrow out of the same
    // storage the creation below adds to.
    const auto prepassColour = fboService.getAttachmentsOfType(*prepassBuffer, FboAttachmentType::Color);
    if (prepassColour.empty())
    {
        return std::unexpected("the ambient occlusion prepass has no colour attachment to reduce");
    }

    // One channel at full precision, and neither half of that is arbitrary. One, because a cell
    // holds a distance and nothing else, and the copy to the CPU transfers whole texels — three
    // unread channels would be three quarters of the transfer. Full precision, because the
    // distances are world units across a city 26,000 of them wide, and a half float quantises to
    // sixteen units out there.
    const auto grid = fboService.create(
        CreateFboDTO{.type = FboType::Planar,
                     .attachments = {CreateFboAttachmentDTO{.width = occlusionGridWidth,
                                                            .height = occlusionGridHeight,
                                                            .type = FboAttachmentType::Color,
                                                            .captureFormat = TextureFormat::R32F,
                                                            .internalFormat = TextureFormat::R32F}}});
    if (!grid)
    {
        return std::unexpected("the occlusion culler has no grid buffer: " + grid.error());
    }

    const auto gridAttachments =
        fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(grid.value()), FboAttachmentType::Color);
    if (gridAttachments.empty())
    {
        return std::unexpected("the occlusion culler's grid buffer has no colour attachment");
    }

    const auto reduce = postProcessService.create(dto.reduceShader, grid.value(), 0, false, "occlusion grid");
    postProcessService.addInput(reduce, prepassColour.front());
    // The alpha this pass writes is not coverage, and a blend would read it as one: every cell is
    // written every frame, and what is in the channel is a distance the CPU is about to read back.
    postProcessService.setBlend(reduce, false);

    camera.occlusionCulling = dto.culling;
    camera.occlusionCulling.enabled = true;
    camera.occlusionCulling.grid = gridAttachments.front();
    camera.occlusionCulling.passes = {reduce};
    // The one state field, cleared here so a DTO cannot seed it: nothing has been read back yet, and
    // the first frames run with no grid at all and therefore with no culling.
    camera.occlusionCulling.readback.reset();

    logger.info("Occlusion culling enabled: a {}x{} grid off the occlusion prepass, {} cell(s) of margin and {:.0f} "
                "unit(s) of travel slack",
                occlusionGridWidth, occlusionGridHeight, camera.occlusionCulling.cellMargin,
                static_cast<double>(camera.occlusionCulling.travelSlack));

    return {};
}

} // namespace raceengine
