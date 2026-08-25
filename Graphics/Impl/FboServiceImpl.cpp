// FboService bodies. Declarations are in Graphics/Services/FboService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <ranges>
#include <string>
#include <vector>

module raceengine.graphics;

import :FboService;
import :IGpuResourceFactory;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

FboService::FboService(MemoryStorageService& memoryStorageService, IGpuResourceFactory& gpuResourceFactory) :
    memoryStorageService(memoryStorageService),
    gpuResourceFactory(gpuResourceFactory)
{
}

std::expected<Resource<Fbo>, std::string> FboService::create(const CreateFboDTO& createFboDTO) const
{
    auto createAttachments = [&](const auto& attachmentsDto)
    {
        std::vector<Resource<FboAttachment>> attachments;

        for (auto& attachment : attachmentsDto)
        {
            attachments.push_back(
                memoryStorageService.bufferAttachments.add(FboAttachment{.type = attachment.type,
                                                                         .width = attachment.width,
                                                                         .height = attachment.height,
                                                                         .captureFormat = attachment.captureFormat,
                                                                         .internalFormat = attachment.internalFormat,
                                                                         .depthComparison = attachment.depthComparison,
                                                                         .levels = attachment.levels}));
        }

        return attachments;
    };

    auto fbo = Fbo{.type = createFboDTO.type, .attachments = createAttachments(createFboDTO.attachments)};

    const auto gpuResourceId = gpuResourceFactory.createFbo(fbo);
    if (!gpuResourceId)
    {
        return std::unexpected(gpuResourceId.error());
    }

    fbo.gpuResourceId = gpuResourceId.value();

    return memoryStorageService.frameBuffers.add(fbo);
}

std::expected<Resource<Fbo>, std::string>
FboService::compose(const std::vector<Resource<FboAttachment>>& attachments) const
{
    for (const auto& attachmentKey : attachments)
    {
        if (memoryStorageService.bufferAttachments.find(attachmentKey) == nullptr)
        {
            return std::unexpected("a composed framebuffer names an attachment that is not live");
        }
    }

    // No factory call on purpose: createFbo creates an image for every attachment it is handed, and
    // these attachments already carry the images their owners created. The record is the whole of
    // what a composed framebuffer is.
    return memoryStorageService.frameBuffers.add(Fbo{.type = FboType::Planar, .attachments = attachments});
}

std::expected<void, std::string> FboService::recreate(const Resource<Fbo>& fbo) const
{
    const auto* stored = memoryStorageService.frameBuffers.find(fbo);
    if (stored == nullptr)
    {
        return std::unexpected("the framebuffer handle names no live framebuffer");
    }

    auto frameBuffer = *stored;

    gpuResourceFactory.deleteFbo(frameBuffer);
    memoryStorageService.frameBuffers.mutate(fbo, [](Fbo& target) { target.gpuResourceId.reset(); });

    const auto gpuResourceId = gpuResourceFactory.createFbo(frameBuffer);
    if (!gpuResourceId)
    {
        return std::unexpected(gpuResourceId.error());
    }

    memoryStorageService.frameBuffers.mutate(fbo, [&](Fbo& target) { target.gpuResourceId = gpuResourceId.value(); });

    return {};
}

std::expected<void, std::string> FboService::resize(const Resource<Fbo>& fbo, unsigned int width,
                                                    unsigned int height) const
{
    const auto* frameBuffer = memoryStorageService.frameBuffers.find(fbo);
    if (frameBuffer == nullptr)
    {
        return std::unexpected("the framebuffer handle names no live framebuffer");
    }

    for (const auto& attachmentKey : frameBuffer->attachments)
    {
        memoryStorageService.bufferAttachments.mutate(attachmentKey,
                                                      [&](FboAttachment& attachment)
                                                      {
                                                          attachment.width = width;
                                                          attachment.height = height;
                                                      });
    }

    return recreate(fbo);
}

std::vector<Resource<FboAttachment>> FboService::getAttachmentsOfType(const Fbo& fbo, FboAttachmentType type) const
{
    auto attachmentsOfType = fbo.attachments | std::views::filter(
                                                   [&](const auto& attachmentKey)
                                                   {
                                                       const auto* attachment =
                                                           memoryStorageService.bufferAttachments.find(attachmentKey);

                                                       return attachment != nullptr && attachment->type == type;
                                                   });

    return std::vector(attachmentsOfType.begin(), attachmentsOfType.end());
}

} // namespace raceengine
