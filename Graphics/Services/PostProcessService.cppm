module;

#include <string>

export module raceengine.graphics:PostProcessService;

import :FboService;
import :Window;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class PostProcessService
{
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    IWindow& window;

public:
    explicit PostProcessService(
        MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window);
    [[nodiscard]] Resource<PostProcess> create(const std::string& id, const Resource<Shader>& shader) const;
    void addInput(const Resource<PostProcess>& postProcess, const Resource<FboAttachment>& attachment) const;
    void recreateOutputBuffer(const Resource<PostProcess> & postProcess, int width, int height) const;
};

PostProcessService::PostProcessService(
    MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window) :
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    window(window)
{

}

Resource<PostProcess> PostProcessService::create(const std::string&, const Resource<Shader>& shader) const
{
    auto state = window.state();
    auto windowWidth = static_cast<unsigned int>(state.windowWidth);
    auto windowHeight = static_cast<unsigned int>(state.windowHeight);

    return memoryStorageService.postProcesses.add(PostProcess{
        .shader = shader,
        .output = fboService.create(
            CreateFboDTO{
                .type = FboType::Planar,
                .attachments = {
                    CreateFboAttachmentDTO{
                        .width = windowWidth,
                        .height = windowHeight,
                        .type = FboAttachmentType::Color,
                        .captureFormat = TextureFormat::RGBA,
                        .internalFormat = TextureFormat::RGBA16F
                    }
                }
            })
    });
}

void PostProcessService::addInput(const Resource<PostProcess>& postProcessKey, const Resource<FboAttachment>& attachment) const
{
    auto postProcess = memoryStorageService.postProcesses.get(postProcessKey);
    postProcess.inputs.push_back(attachment);

    memoryStorageService.postProcesses.update(postProcessKey, postProcess);
}

void PostProcessService::recreateOutputBuffer(const Resource<PostProcess>& postProcessKey, int width, int height) const
{
    auto postProcess = memoryStorageService.postProcesses.get(postProcessKey);

    if (postProcess.output.has_value())
    {
        fboService.resize(postProcess.output.value(), static_cast<unsigned int>(width),
                          static_cast<unsigned int>(height));
        memoryStorageService.postProcesses.update(postProcessKey, postProcess);
    }
}

} // namespace raceengine
