module;

#include <expected>
#include <string>

export module raceengine.graphics:IGpuResourceFactory;

import raceengine.graphics.models;

namespace raceengine
{

// Which family of GPU object an id belongs to. GL needs it because a name is only meaningful
// against the object type it was generated for — buffer 3, texture 3 and framebuffer 3 are three
// different objects — so the kind is what picks the right glDelete*. Vulkan hands out one
// monotonic id space and could get by without it, which is exactly why it is the *caller's*
// knowledge that is encoded here rather than the backend's.
export enum class GpuResourceKind {
    Buffer,
    Texture,
    CubeMap,
    VertexArray,
    ShaderProgram,
    FrameBuffer,
};

// What a backend still owns. This is not a debug hook: releasing is the half of ownership that
// has no other observable effect, so a count is the only way a release is seen to have happened
// and the only way a leak is seen at all.
export struct GpuResourceCensus
{
    unsigned int buffers{};
    unsigned int textures{};
    unsigned int vertexArrays{};
    unsigned int shaderPrograms{};
    unsigned int frameBuffers{};

    [[nodiscard]] unsigned int total() const
    {
        return buffers + textures + vertexArrays + shaderPrograms + frameBuffers;
    }
};

// GPU objects built from the engine's own model types. Nothing here belongs to a frame: a
// factory call runs at load time, from a resize callback or from an unload, never between two
// passes, and the services that hold this reference (FboService, ShaderService, CubeMapService,
// AssetService) record nothing. Ids are opaque unsigned ints, never API handle types.
//
// Every creation is fallible for a reason the caller can act on — a framebuffer the driver
// reports incomplete, a program that will not link, an attachment format the backend has no
// equivalent for — so each reports it rather than returning a bare id and throwing from
// inside. The release calls cannot fail: an id the backend does not own is simply not its, and a
// backend that is already torn down has nothing to do.
export class IGpuResourceFactory
{
public:
    IGpuResourceFactory() = default;
    IGpuResourceFactory(const IGpuResourceFactory&) = delete;
    IGpuResourceFactory(IGpuResourceFactory&&) = delete;
    IGpuResourceFactory& operator=(const IGpuResourceFactory&) = delete;
    IGpuResourceFactory& operator=(IGpuResourceFactory&&) = delete;
    virtual ~IGpuResourceFactory() = default;

    [[nodiscard]] virtual std::expected<unsigned int, std::string>
    createShaderObject(const ShaderDescriptor& shaderDescriptor) = 0;
    [[nodiscard]] virtual std::expected<unsigned int, std::string>
    createCubeMap(const Texture& front, const Texture& back, const Texture& left, const Texture& right,
                  const Texture& top, const Texture& bottom) = 0;
    [[nodiscard]] virtual std::expected<unsigned int, std::string> createFbo(const Fbo& fbo) = 0;
    virtual void deleteFbo(Fbo& fbo) = 0;

    // Releases one GPU object an engine model was holding the id of. The caller's guard against a
    // double release is the handle: an id is read out of a live element and the element is
    // removed in the same step, so the same id is never presented twice.
    //
    // "Released" does not mean "destroyed now" — on Vulkan two frames are in flight and an image
    // the GPU may still be sampling cannot be destroyed at the call. What it means is that the
    // engine will never name this id again and the backend owns the timing.
    virtual void releaseGpuResource(GpuResourceKind kind, unsigned int gpuResourceId) = 0;

    // Drops whatever a backend keyed on a *handle* rather than on an id stored in the model.
    // Vulkan caches a uniform buffer and a descriptor set per (material, environment) pair
    // because material contents are load-time constant; GL keeps a "currently bound" latch. Both
    // are keyed by the material's slot, so both have to go when the slot does — otherwise the
    // next material to land in that slot inherits the previous one's descriptors.
    virtual void releaseMaterial(const Resource<Material>& material) = 0;

    [[nodiscard]] virtual GpuResourceCensus gpuResourceCensus() const = 0;
};

} // namespace raceengine
