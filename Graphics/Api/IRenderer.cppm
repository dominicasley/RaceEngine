module;

#include <optional>
#include <string>

export module raceengine.graphics:IRenderer;

import raceengine.graphics.models;

namespace raceengine
{

// Backend-neutral renderer surface: exactly the methods the composition root and the
// graphics services call. Signatures carry only engine-model types and primitives —
// GPU object ids are opaque unsigned ints, never API handle types.
export class IRenderer
{
public:
    IRenderer() = default;
    IRenderer(const IRenderer&) = delete;
    IRenderer(IRenderer&&) = delete;
    IRenderer& operator=(const IRenderer&) = delete;
    IRenderer& operator=(IRenderer&&) = delete;
    virtual ~IRenderer() = default;

    virtual bool init() = 0;
    virtual void draw(Scene& scene, Camera& camera, float delta) = 0;
    virtual void drawFullScreenQuad(const Resource<Shader>& shader,
                                    const Resource<FboAttachment>& attachment) const = 0;
    virtual void setViewport(int width, int height) = 0;
    virtual std::optional<unsigned int> createShaderObject(const ShaderDescriptor& shaderDescriptor) = 0;
    [[nodiscard]] virtual unsigned int createCubeMap(const Texture& front, const Texture& back, const Texture& left,
                                                     const Texture& right, const Texture& top,
                                                     const Texture& bottom) const = 0;
    [[nodiscard]] virtual unsigned int createFbo(const Fbo& fbo) const = 0;
    virtual void deleteFbo(Fbo& fbo) const = 0;
    virtual void captureFrame(const std::string& path) = 0;
};

} // namespace raceengine
