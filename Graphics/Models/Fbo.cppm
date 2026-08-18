module;

#include <optional>
#include <vector>

export module raceengine.graphics.models:Fbo;

import raceengine.resource;
import :Shader;
import :Texture;

namespace raceengine
{

export enum class FboAttachmentType { Color, Depth, Stencil };

export enum class FboType { Planar, CubeMap };

// Hardware depth comparison: the sampler returns the result of comparing the reference value
// against the stored depth rather than the depth itself, filtered — GL's
// GL_TEXTURE_COMPARE_MODE/GL_TEXTURE_COMPARE_FUNC and a Vulkan sampler's compareEnable/compareOp.
// It is the sampler2DShadow path percentage-closer filtering needs, and it is stated on the
// attachment because the sampler belongs to the image on one backend and to the texture object on
// the other; the engine has no separate sampler model to hang it on. Meaningful only on a Depth
// attachment; both backends ignore it elsewhere.
export enum class DepthComparison { None, LessOrEqual };

export struct FboAttachment
{
    FboAttachmentType type;
    std::optional<unsigned int> gpuResourceId{};
    unsigned int width;
    unsigned int height;
    TextureFormat captureFormat;
    TextureFormat internalFormat;
    DepthComparison depthComparison = DepthComparison::None;
    std::vector<unsigned char> data{};
};

export struct Fbo
{
    FboType type;
    std::optional<unsigned int> gpuResourceId{};
    std::vector<Resource<FboAttachment>> attachments;
};

export struct PostProcess
{
    Resource<Shader> shader;
    std::optional<Resource<Fbo>> output;
    std::vector<Resource<FboAttachment>> inputs{};
    std::vector<Resource<FboAttachment>> attachments{};
};

export struct Presenter
{
    Resource<FboAttachment> output;
    Resource<Shader> shader;
};

} // namespace raceengine
