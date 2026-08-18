module;

#include <expected>
#include <ranges>
#include <string>
#include <vector>

export module raceengine.graphics:FboService;

import :IGpuResourceFactory;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class FboService
{
private:
    MemoryStorageService& memoryStorageService;
    IGpuResourceFactory& gpuResourceFactory;

public:
    explicit FboService(MemoryStorageService& memoryStorageService, IGpuResourceFactory& gpuResourceFactory);
    [[nodiscard]] std::expected<Resource<Fbo>, std::string> create(const CreateFboDTO& createFboDTO) const;
    [[nodiscard]] std::expected<void, std::string> recreate(const Resource<Fbo>& fbo) const;
    [[nodiscard]] std::expected<void, std::string> resize(const Resource<Fbo>& fbo, unsigned int width,
                                                          unsigned int height) const;

    [[nodiscard]] std::vector<Resource<FboAttachment>> getAttachmentsOfType(const Fbo& fbo,
                                                                            FboAttachmentType type) const;
};

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
                                                                         .internalFormat = attachment.internalFormat}));
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

// A framebuffer that fails to come back is left with no GPU id rather than the stale one the
// delete just invalidated: whoever handles the error is choosing what to do about a buffer
// that cannot be drawn into, not about one that can still be drawn into by accident.
//
// The framebuffer is copied out rather than mutated in place because deleteFbo/createFbo take it
// by reference and reach into the attachment storage themselves; only the resulting id is written
// back through the element, and it is written through mutate(), which touches that one field.
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
