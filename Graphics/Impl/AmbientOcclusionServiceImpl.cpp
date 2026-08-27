// AmbientOcclusionService bodies. Declarations are in Graphics/Services/AmbientOcclusionService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <expected>
#include <string>

#include <glm/glm.hpp>
#include <spdlog/logger.h>

module raceengine.graphics;

import :AmbientOcclusionService;
import :FboService;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

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

    // The gather and its blur run at half resolution: occlusion is a low-frequency term, the
    // horizon search is the frame's most expensive fullscreen pass, and the upsample below puts
    // the result back at the view's size without letting it cross a depth edge. The prepass stays
    // full resolution — it is the geometry every consumer measures against, including the upsample.
    const auto halfWidth = std::max(width / 2u, 1u);
    const auto halfHeight = std::max(height / 2u, 1u);

    // A visibility term is one number between zero and one, so one eight-bit channel states it
    // exactly as well as four float ones would.
    const auto gatherTarget =
        fboService.create(CreateFboDTO{.type = FboType::Planar,
                                       .attachments = {CreateFboAttachmentDTO{.width = halfWidth,
                                                                              .height = halfHeight,
                                                                              .type = FboAttachmentType::Color,
                                                                              .captureFormat = TextureFormat::R,
                                                                              .internalFormat = TextureFormat::R}}});
    if (!gatherTarget)
    {
        return std::unexpected("the occlusion gather has no target: " + gatherTarget.error());
    }

    const auto blurTarget =
        fboService.create(CreateFboDTO{.type = FboType::Planar,
                                       .attachments = {CreateFboAttachmentDTO{.width = halfWidth,
                                                                              .height = halfHeight,
                                                                              .type = FboAttachmentType::Color,
                                                                              .captureFormat = TextureFormat::R,
                                                                              .internalFormat = TextureFormat::R}}});
    if (!blurTarget)
    {
        return std::unexpected("the occlusion blur has no target: " + blurTarget.error());
    }

    const auto upsampleTarget =
        fboService.create(CreateFboDTO{.type = FboType::Planar,
                                       .attachments = {CreateFboAttachmentDTO{.width = width,
                                                                              .height = height,
                                                                              .type = FboAttachmentType::Color,
                                                                              .captureFormat = TextureFormat::R,
                                                                              .internalFormat = TextureFormat::R}}});
    if (!upsampleTarget)
    {
        return std::unexpected("the occlusion upsample has no target: " + upsampleTarget.error());
    }

    const auto prepassColour = fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(prepass.value()),
                                                               FboAttachmentType::Color);
    const auto gathered = fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(gatherTarget.value()),
                                                          FboAttachmentType::Color);
    const auto blurred = fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(blurTarget.value()),
                                                         FboAttachmentType::Color);
    const auto upsampled = fboService.getAttachmentsOfType(
        memoryStorageService.frameBuffers.get(upsampleTarget.value()), FboAttachmentType::Color);
    if (prepassColour.empty() || gathered.empty() || blurred.empty() || upsampled.empty())
    {
        return std::unexpected("one of the occlusion buffers came back without a colour attachment");
    }

    const auto gather = postProcessService.create(dto.gatherShader, gatherTarget.value(), 0, false, "gtao gather");
    postProcessService.addInput(gather, prepassColour.front());
    // The gather is the only pass here with numbers of its own, and they are its rather than the
    // camera's: the buffers are sized once and the search is tuned once.
    postProcessService.setParameters(gather, glm::vec4(dto.occlusion.radius, dto.occlusion.strength, 0.0f, 0.0f));

    // Two inputs, which is the capability this step was waiting on: the term being smoothed, and the
    // geometry that says where smoothing it would cross an edge.
    const auto blur = postProcessService.create(dto.blurShader, blurTarget.value(), 0, false, "gtao blur");
    postProcessService.addInput(blur, gathered.front());
    postProcessService.addInput(blur, prepassColour.front());

    // Back to the view's own size, against the geometry: the term being lifted, and the prepass
    // that says which of the four half-resolution texels under a pixel share its surface.
    const auto upsample =
        postProcessService.create(dto.upsampleShader, upsampleTarget.value(), 0, false, "gtao upsample");
    postProcessService.addInput(upsample, blurred.front());
    postProcessService.addInput(upsample, prepassColour.front());

    camera.ambientOcclusion = dto.occlusion;
    camera.ambientOcclusion.enabled = true;
    camera.ambientOcclusion.prepass = prepass.value();
    camera.ambientOcclusion.prepassShader = dto.prepassShader;
    camera.ambientOcclusion.occlusion = upsampled.front();
    camera.ambientOcclusion.passes = {gather, blur, upsample};

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
    // writes, and a second list of the same buffers would be a second thing to keep in step. The
    // list order is this service's own — gather, blur, upsample — and the first two run at half
    // the view's size, which is the same split enable() built them with.
    for (size_t index = 0; index < occlusion.passes.size(); index++)
    {
        const auto* pass = memoryStorageService.postProcesses.find(occlusion.passes[index]);
        if (pass == nullptr || !pass->output.has_value())
        {
            return std::unexpected("an occlusion pass no longer names a buffer to rebuild");
        }

        const auto halfSized = index < 2;
        const auto passWidth = halfSized ? std::max(newWidth / 2u, 1u) : newWidth;
        const auto passHeight = halfSized ? std::max(newHeight / 2u, 1u) : newHeight;
        if (const auto resized = fboService.resize(pass->output.value(), passWidth, passHeight); !resized)
        {
            return std::unexpected("an occlusion buffer was not rebuilt: " + resized.error());
        }
    }

    return {};
}

} // namespace raceengine
