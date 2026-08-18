module;

#include <expected>
#include <string>

export module raceengine.graphics:IGpuResourceFactory;

import raceengine.graphics.models;

namespace raceengine
{

// GPU objects built from the engine's own model types. Nothing here belongs to a frame: a
// factory call runs at load time or from a resize callback, never between two passes, and
// the three services that hold this reference (FboService, ShaderService, CubeMapService)
// record nothing. Ids are opaque unsigned ints, never API handle types.
//
// Every creation is fallible for a reason the caller can act on — a framebuffer the driver
// reports incomplete, a program that will not link, an attachment format the backend has no
// equivalent for — so each reports it rather than returning a bare id and throwing from
// inside. deleteFbo releases what createFbo made and cannot fail.
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
};

} // namespace raceengine
