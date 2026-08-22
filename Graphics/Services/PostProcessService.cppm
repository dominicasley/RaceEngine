module;

#include <expected>
#include <string>

#include <glm/glm.hpp>

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
    // A pass with an output buffer of its own: one colour attachment at the window's size, with
    // `levels` mips. One level is a plain post-process target; more makes it a chain, and the
    // passes that walk it are created through the overload below so they share this one buffer.
    [[nodiscard]] std::expected<Resource<PostProcess>, std::string>
    create(const std::string& id, const Resource<Shader>& shader, unsigned int levels = 1) const;
    // A pass that renders into a framebuffer somebody else owns — the level of a chain, or a
    // second pass over one target. Nothing here can fail: the buffer already exists. It does not
    // follow the window by default for the reason PostProcess::tracksWindowSize gives: the owner
    // decided that buffer's size and is the one who can rebuild it.
    [[nodiscard]] Resource<PostProcess> create(const Resource<Shader>& shader, const Resource<Fbo>& output,
                                               unsigned int outputLevel = 0, bool tracksWindowSize = false) const;
    void addInput(const Resource<PostProcess>& postProcess, const Resource<FboAttachment>& attachment,
                  unsigned int level = 0) const;
    // The four numbers this pass's shader reads out of the fullscreen push constant. See
    // PostProcess::parameters for why they belong to the pass.
    void setParameters(const Resource<PostProcess>& postProcess, const glm::vec4& parameters) const;
    [[nodiscard]] std::expected<void, std::string> recreateOutputBuffer(const Resource<PostProcess>& postProcess,
                                                                        int width, int height) const;
};

} // namespace raceengine
