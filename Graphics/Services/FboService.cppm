module;

#include <ranges>
#include <vector>

export module raceengine.graphics:FboService;

import :OpenGLRenderer;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class FboService
{
private:
    MemoryStorageService& memoryStorageService;
    OpenGLRenderer& renderer;

public:
    explicit FboService(MemoryStorageService& memoryStorageService, OpenGLRenderer& renderer);
    [[nodiscard]] Resource<Fbo> create(const CreateFboDTO& createFboDTO) const;
    void recreate(const Resource<Fbo>& fbo) const;
    void resize(const Resource<Fbo>& fbo, unsigned int width, unsigned int height) const;

    [[nodiscard]] std::vector<Resource<FboAttachment>> getAttachmentsOfType(const Fbo& fbo,
                                                                            FboAttachmentType type) const;
};

FboService::FboService(MemoryStorageService& memoryStorageService, OpenGLRenderer& renderer) :
    memoryStorageService(memoryStorageService),
    renderer(renderer)
{
}

Resource<Fbo> FboService::create(const CreateFboDTO& createFboDTO) const
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

    fbo.gpuResourceId = renderer.createFbo(fbo);

    return memoryStorageService.frameBuffers.add(fbo);
}

void FboService::recreate(const Resource<Fbo>& fbo) const
{
    auto frameBuffer = memoryStorageService.frameBuffers.get(fbo);

    renderer.deleteFbo(frameBuffer);
    frameBuffer.gpuResourceId = renderer.createFbo(frameBuffer);

    memoryStorageService.frameBuffers.update(fbo, frameBuffer);
}

void FboService::resize(const Resource<Fbo>& fbo, unsigned int width, unsigned int height) const
{
    auto frameBuffer = memoryStorageService.frameBuffers.get(fbo);

    for (auto& attachmentKey : frameBuffer.attachments)
    {
        auto attachment = memoryStorageService.bufferAttachments.get(attachmentKey);

        attachment.width = width;
        attachment.height = height;

        memoryStorageService.bufferAttachments.update(attachmentKey, attachment);
    }

    recreate(fbo);
}

std::vector<Resource<FboAttachment>> FboService::getAttachmentsOfType(const Fbo& fbo, FboAttachmentType type) const
{
    auto attachmentsOfType =
        fbo.attachments |
        std::views::filter([&](const auto& attachment)
                           { return memoryStorageService.bufferAttachments.get(attachment).type == type; });

    return std::vector(attachmentsOfType.begin(), attachmentsOfType.end());
}

} // namespace raceengine
