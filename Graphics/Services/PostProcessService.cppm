module;

#include <expected>
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
    explicit PostProcessService(MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window);
    [[nodiscard]] std::expected<Resource<PostProcess>, std::string> create(const std::string& id,
                                                                           const Resource<Shader>& shader) const;
    void addInput(const Resource<PostProcess>& postProcess, const Resource<FboAttachment>& attachment) const;
    [[nodiscard]] std::expected<void, std::string> recreateOutputBuffer(const Resource<PostProcess>& postProcess,
                                                                        int width, int height) const;
};

PostProcessService::PostProcessService(MemoryStorageService& memoryStorageService, FboService& fboService,
                                       IWindow& window) :
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    window(window)
{
}

std::expected<Resource<PostProcess>, std::string> PostProcessService::create(const std::string& id,
                                                                             const Resource<Shader>& shader) const
{
    auto state = window.state();
    auto windowWidth = static_cast<unsigned int>(state.windowWidth);
    auto windowHeight = static_cast<unsigned int>(state.windowHeight);

    const auto output = fboService.create(
        CreateFboDTO{.type = FboType::Planar,
                     .attachments = {CreateFboAttachmentDTO{.width = windowWidth,
                                                            .height = windowHeight,
                                                            .type = FboAttachmentType::Color,
                                                            .captureFormat = TextureFormat::RGBA,
                                                            .internalFormat = TextureFormat::RGBA16F}}});

    if (!output)
    {
        return std::unexpected("post process '" + id + "' has no output buffer: " + output.error());
    }

    return memoryStorageService.postProcesses.add(PostProcess{.shader = shader, .output = output.value()});
}

void PostProcessService::addInput(const Resource<PostProcess>& postProcessKey,
                                  const Resource<FboAttachment>& attachment) const
{
    memoryStorageService.postProcesses.mutate(postProcessKey, [&](PostProcess& postProcess)
                                              { postProcess.inputs.push_back(attachment); });
}

std::expected<void, std::string> PostProcessService::recreateOutputBuffer(const Resource<PostProcess>& postProcessKey,
                                                                          int width, int height) const
{
    const auto* postProcess = memoryStorageService.postProcesses.find(postProcessKey);
    if (postProcess == nullptr || !postProcess->output.has_value())
    {
        return {};
    }

    // The post-process element itself is unchanged by a resize — only the framebuffer it names
    // is rebuilt — so there is nothing to write back here. The write-back this used to do was a
    // copy of an untouched element assigned over itself.
    return fboService.resize(postProcess->output.value(), static_cast<unsigned int>(width),
                             static_cast<unsigned int>(height));
}

} // namespace raceengine
