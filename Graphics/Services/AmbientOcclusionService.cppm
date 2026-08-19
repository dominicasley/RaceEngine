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

AmbientOcclusionService::AmbientOcclusionService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                                                 FboService& fboService, PostProcessService& postProcessService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    postProcessService(postProcessService)
{
}

std::expected<void, std::string> AmbientOcclusionService::enable(Camera& camera,
                                                                 const CreateAmbientOcclusionDTO& dto) const
{
    if (!camera.output.has_value())
    {
        return std::unexpected("a camera with no render target has no view to gather occlusion from");
    }

    const auto* cameraBuffer = memoryStorageService.frameBuffers.find(camera.output.value());
    if (cameraBuffer == nullptr)
    {
        return std::unexpected("the camera's render target handle names no live framebuffer");
    }

    // Resolved before anything is created, for the reason AutoExposureService resolves first: this
    // is a borrow out of the storage the creations below add to.
    const auto sceneColour = fboService.getAttachmentsOfType(*cameraBuffer, FboAttachmentType::Color);
    if (sceneColour.empty())
    {
        return std::unexpected("the camera's render target has no colour attachment to size the gather by");
    }

    const auto* sizeSource = memoryStorageService.bufferAttachments.find(sceneColour.front());
    if (sizeSource == nullptr)
    {
        return std::unexpected("the camera's colour attachment is no longer loaded");
    }

    const auto width = sizeSource->width;
    const auto height = sizeSource->height;

    // Normal in rgb and view depth in a, at half precision: the normal is a direction and the depth
    // is a distance the gather compares against a world radius, and neither survives eight bits.
    // The depth attachment beside it is never sampled — it is what makes the prepass draw the
    // nearest surface rather than the last one submitted.
    const auto prepass = fboService.create(
        CreateFboDTO{.type = FboType::Planar,
                     .attachments = {CreateFboAttachmentDTO{.width = width,
                                                            .height = height,
                                                            .type = FboAttachmentType::Color,
                                                            .captureFormat = TextureFormat::RGBA,
                                                            .internalFormat = TextureFormat::RGBA16F},
                                     CreateFboAttachmentDTO{.width = width,
                                                            .height = height,
                                                            .type = FboAttachmentType::Depth,
                                                            .captureFormat = TextureFormat::DepthComponent,
                                                            .internalFormat = TextureFormat::DepthComponent32F}}});
    if (!prepass)
    {
        return std::unexpected("the occlusion prepass has no buffer: " + prepass.error());
    }

    // A visibility term is one number between zero and one, so one eight-bit channel states it
    // exactly as well as four float ones would.
    const auto gatherTarget = fboService.create(
        CreateFboDTO{.type = FboType::Planar,
                     .attachments = {CreateFboAttachmentDTO{.width = width,
                                                            .height = height,
                                                            .type = FboAttachmentType::Color,
                                                            .captureFormat = TextureFormat::R,
                                                            .internalFormat = TextureFormat::R}}});
    if (!gatherTarget)
    {
        return std::unexpected("the occlusion gather has no target: " + gatherTarget.error());
    }

    const auto blurTarget = fboService.create(
        CreateFboDTO{.type = FboType::Planar,
                     .attachments = {CreateFboAttachmentDTO{.width = width,
                                                            .height = height,
                                                            .type = FboAttachmentType::Color,
                                                            .captureFormat = TextureFormat::R,
                                                            .internalFormat = TextureFormat::R}}});
    if (!blurTarget)
    {
        return std::unexpected("the occlusion blur has no target: " + blurTarget.error());
    }

    const auto prepassColour =
        fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(prepass.value()), FboAttachmentType::Color);
    const auto gathered = fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(gatherTarget.value()),
                                                          FboAttachmentType::Color);
    const auto blurred = fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(blurTarget.value()),
                                                         FboAttachmentType::Color);
    if (prepassColour.empty() || gathered.empty() || blurred.empty())
    {
        return std::unexpected("one of the occlusion buffers came back without a colour attachment");
    }

    const auto gather = postProcessService.create(dto.gatherShader, gatherTarget.value());
    postProcessService.addInput(gather, prepassColour.front());
    // The gather is the only pass here with numbers of its own, and they are its rather than the
    // camera's: the buffers are sized once and the search is tuned once.
    postProcessService.setParameters(gather,
                                     glm::vec4(dto.occlusion.radius, dto.occlusion.strength, 0.0f, 0.0f));

    // Two inputs, which is the capability this step was waiting on: the term being smoothed, and the
    // geometry that says where smoothing it would cross an edge.
    const auto blur = postProcessService.create(dto.blurShader, blurTarget.value());
    postProcessService.addInput(blur, gathered.front());
    postProcessService.addInput(blur, prepassColour.front());

    camera.ambientOcclusion = dto.occlusion;
    camera.ambientOcclusion.enabled = true;
    camera.ambientOcclusion.prepass = prepass.value();
    camera.ambientOcclusion.prepassShader = dto.prepassShader;
    camera.ambientOcclusion.occlusion = blurred.front();
    camera.ambientOcclusion.passes = {gather, blur};

    logger.info("Ambient occlusion enabled: {}x{} prepass, {} slice(s) of {} step(s) over a {:.0f} unit radius, "
                "strength {:.2f}",
                width, height, gtaoSliceCount, gtaoStepsPerSlice, camera.ambientOcclusion.radius,
                camera.ambientOcclusion.strength);

    return {};
}

std::expected<void, std::string> AmbientOcclusionService::resize(const Camera& camera, const int width,
                                                                 const int height) const
{
    const auto& occlusion = camera.ambientOcclusion;

    if (!occlusion.enabled || !occlusion.prepass.has_value())
    {
        return {};
    }

    const auto newWidth = static_cast<unsigned int>(std::max(width, 1));
    const auto newHeight = static_cast<unsigned int>(std::max(height, 1));

    if (const auto resized = fboService.resize(occlusion.prepass.value(), newWidth, newHeight); !resized)
    {
        return std::unexpected("the occlusion prepass was not rebuilt: " + resized.error());
    }

    // Through the passes rather than through handles held beside them: a pass names the buffer it
    // writes, and a second list of the same buffers would be a second thing to keep in step.
    for (const auto& passKey : occlusion.passes)
    {
        const auto* pass = memoryStorageService.postProcesses.find(passKey);
        if (pass == nullptr || !pass->output.has_value())
        {
            return std::unexpected("an occlusion pass no longer names a buffer to rebuild");
        }

        if (const auto resized = fboService.resize(pass->output.value(), newWidth, newHeight); !resized)
        {
            return std::unexpected("an occlusion buffer was not rebuilt: " + resized.error());
        }
    }

    return {};
}

} // namespace raceengine
