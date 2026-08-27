// PostProcessService bodies. Declarations are in Graphics/Services/PostProcessService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <optional>
#include <string>

#include <glm/glm.hpp>

module raceengine.graphics;

import :PostProcessService;
import :FboService;
import :Window;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

PostProcessService::PostProcessService(MemoryStorageService& memoryStorageService, FboService& fboService,
                                       IWindow& window) :
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    window(window)
{
}

std::expected<Resource<PostProcess>, std::string>
PostProcessService::create(const std::string& id, const Resource<Shader>& shader, const unsigned int levels) const
{
    auto state = window.state();
    auto windowWidth = static_cast<unsigned int>(state.windowWidth);
    auto windowHeight = static_cast<unsigned int>(state.windowHeight);

    const auto output =
        fboService.create(CreateFboDTO{.type = FboType::Planar,
                                       .attachments = {CreateFboAttachmentDTO{.width = windowWidth,
                                                                              .height = windowHeight,
                                                                              .type = FboAttachmentType::Color,
                                                                              .captureFormat = TextureFormat::RGBA,
                                                                              .internalFormat = TextureFormat::RGBA16F,
                                                                              .levels = levels}}});

    if (!output)
    {
        return std::unexpected("post process '" + id + "' has no output buffer: " + output.error());
    }

    return memoryStorageService.postProcesses.add(
        PostProcess{.shader = shader, .output = output.value(), .debugName = id});
}

Resource<PostProcess> PostProcessService::create(const Resource<Shader>& shader, const Resource<Fbo>& output,
                                                 const unsigned int outputLevel, const bool tracksWindowSize,
                                                 const std::string& id) const
{
    return memoryStorageService.postProcesses.add(PostProcess{.shader = shader,
                                                              .output = output,
                                                              .debugName = id,
                                                              .outputLevel = outputLevel,
                                                              .tracksWindowSize = tracksWindowSize});
}

void PostProcessService::setBlend(const Resource<PostProcess>& postProcessKey, const bool blend) const
{
    memoryStorageService.postProcesses.mutate(postProcessKey,
                                              [&](PostProcess& postProcess) { postProcess.blend = blend; });
}

void PostProcessService::setBlendWeight(const Resource<PostProcess>& postProcessKey,
                                        const std::optional<float> weight) const
{
    memoryStorageService.postProcesses.mutate(postProcessKey,
                                              [&](PostProcess& postProcess) { postProcess.blendWeight = weight; });
}

void PostProcessService::setLoadColour(const Resource<PostProcess>& postProcessKey, const bool loadColour) const
{
    memoryStorageService.postProcesses.mutate(postProcessKey,
                                              [&](PostProcess& postProcess) { postProcess.loadColour = loadColour; });
}

void PostProcessService::setContentsHeld(const Resource<PostProcess>& postProcessKey, const bool held) const
{
    memoryStorageService.postProcesses.mutate(postProcessKey,
                                              [&](PostProcess& postProcess) { postProcess.contentsHeld = held; });
}

void PostProcessService::setSlice(const Resource<PostProcess>& postProcessKey, const unsigned int slice,
                                  const unsigned int sliceCount) const
{
    const auto count = sliceCount == 0 ? 1u : sliceCount;

    memoryStorageService.postProcesses.mutate(postProcessKey,
                                              [&](PostProcess& postProcess)
                                              {
                                                  postProcess.sliceCount = count;
                                                  postProcess.slice = slice % count;
                                              });
}

void PostProcessService::setWindowSizeDivisor(const Resource<PostProcess>& postProcessKey,
                                              const unsigned int divisor) const
{
    memoryStorageService.postProcesses.mutate(
        postProcessKey, [&](PostProcess& postProcess) { postProcess.windowSizeDivisor = divisor == 0 ? 1u : divisor; });
}

void PostProcessService::addInput(const Resource<PostProcess>& postProcessKey,
                                  const Resource<FboAttachment>& attachment, const unsigned int level) const
{
    memoryStorageService.postProcesses.mutate(
        postProcessKey, [&](PostProcess& postProcess)
        { postProcess.inputs.push_back(PostProcessInput{.attachment = attachment, .level = level}); });
}

void PostProcessService::addInput(const Resource<PostProcess>& postProcessKey, const Resource<Texture>& texture) const
{
    // The input array is sampler2D the whole way across, and a 3D image bound to a 2D slot is
    // undefined with no validation to report it — the descriptor is written, the sample just reads
    // garbage. Refused here, where the mistake is a one-line fix, rather than rendered as a wrong
    // picture three passes downstream.
    if (const auto* stored = memoryStorageService.textures.find(texture); stored != nullptr && stored->depth > 1)
    {
        fail("a texture with depth above one is a volume and cannot ride the fullscreen input array; "
             "use addVolumeInput");
    }

    memoryStorageService.postProcesses.mutate(
        postProcessKey,
        [&](PostProcess& postProcess) { postProcess.inputs.push_back(PostProcessInput{.texture = texture}); });
}

void PostProcessService::addVolumeInput(const Resource<PostProcess>& postProcessKey,
                                        const Resource<Texture>& texture) const
{
    // The inverse of addInput's guard: a flat image where the shader declares a sampler3D is the
    // same undefined bind from the other side.
    if (const auto* stored = memoryStorageService.textures.find(texture); stored != nullptr && stored->depth <= 1)
    {
        fail("addVolumeInput takes a texture with depth above one; use addInput for a picture");
    }

    memoryStorageService.postProcesses.mutate(postProcessKey, [&](PostProcess& postProcess)
                                              { postProcess.volumes.push_back(texture); });
}

void PostProcessService::setParameters(const Resource<PostProcess>& postProcessKey, const glm::vec4& parameters) const
{
    memoryStorageService.postProcesses.mutate(postProcessKey,
                                              [&](PostProcess& postProcess) { postProcess.parameters = parameters; });
}

std::expected<void, std::string> PostProcessService::recreateOutputBuffer(const Resource<PostProcess>& postProcessKey,
                                                                          int width, int height) const
{
    const auto* postProcess = memoryStorageService.postProcesses.find(postProcessKey);
    if (postProcess == nullptr)
    {
        // Reported, not shrugged off: this used to return success for a handle it could not even
        // resolve, so a camera still holding an unloaded post-process resized silently and drew
        // through a buffer nobody rebuilt.
        return std::unexpected("the post-process handle names no live post-process");
    }

    // A post-process with no output of its own writes the default framebuffer, which the window
    // resized already. Nothing to do is not a failure.
    if (!postProcess->output.has_value())
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
