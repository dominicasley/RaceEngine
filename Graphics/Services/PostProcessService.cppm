module;

#include <expected>
#include <optional>
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
                                               unsigned int outputLevel = 0, bool tracksWindowSize = false,
                                               const std::string& id = {}) const;
    void addInput(const Resource<PostProcess>& postProcess, const Resource<FboAttachment>& attachment,
                  unsigned int level = 0) const;
    // An input this frame did not draw: a picture, uploaded once and sampled every frame after. The
    // overload exists rather than a flag because the two cannot be confused at a call site — a pass
    // reading the bloom chain and a pass reading a lens dirt plate are naming different kinds of
    // thing, and the backend treats them differently (a texture has no layout to move and no level
    // to choose). Order still decides the element of the shader's sampler array, exactly as above.
    void addInput(const Resource<PostProcess>& postProcess, const Resource<Texture>& texture) const;
    // A three-dimensional texture this pass samples — baked volumetric noise. Its own call rather
    // than an overload resolved by the texture's shape because the destination differs: volumes
    // bind the fullscreen set's sampler3D slots beside the grade, never the sampler2D input array,
    // and a volume through addInput would bind a 3D image to a 2D slot with nothing to say so —
    // which is why both calls check the texture's depth and refuse the wrong kind outright. Order
    // decides the slot, exactly as it decides an input's element.
    void addVolumeInput(const Resource<PostProcess>& postProcess, const Resource<Texture>& texture) const;
    // The four numbers this pass's shader reads out of the fullscreen push constant. See
    // PostProcess::parameters for why they belong to the pass.
    void setParameters(const Resource<PostProcess>& postProcess, const glm::vec4& parameters) const;
    // See PostProcess::blend. Off is for a pass whose alpha channel is data, not coverage.
    void setBlend(const Resource<PostProcess>& postProcess, bool blend) const;
    // See PostProcess::blendWeight. A stated mix for a pass whose alpha is already carrying
    // something else; unset puts it back on its own alpha.
    void setBlendWeight(const Resource<PostProcess>& postProcess, std::optional<float> weight) const;
    // See PostProcess::loadColour. True for a pass that reads its target — because it blends
    // against what is there, or because it writes only part of it.
    void setLoadColour(const Resource<PostProcess>& postProcess, bool loadColour) const;
    // See PostProcess::contentsHeld. The frame skips a held pass entirely and leaves its target
    // exactly as it stands.
    void setContentsHeld(const Resource<PostProcess>& postProcess, bool held) const;
    // See PostProcess::sliceCount. Which vertical strip of its target this pass writes next, of
    // how many; a count of one puts it back to writing the whole target. The slice is taken modulo
    // the count, so a caller may hand it a frame counter and never think about wrapping.
    void setSlice(const Resource<PostProcess>& postProcess, unsigned int slice, unsigned int sliceCount) const;
    // See PostProcess::windowSizeDivisor.
    void setWindowSizeDivisor(const Resource<PostProcess>& postProcess, unsigned int divisor) const;
    [[nodiscard]] std::expected<void, std::string> recreateOutputBuffer(const Resource<PostProcess>& postProcess,
                                                                        int width, int height) const;
};

} // namespace raceengine
