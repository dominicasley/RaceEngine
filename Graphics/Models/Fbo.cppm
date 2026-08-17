module;

#include <optional>
#include <vector>

export module raceengine.graphics.models:Fbo;

import raceengine.resource;
import :Shader;
import :Texture;

namespace raceengine
{

export enum class FboAttachmentType
{
    Color,
    Depth,
    Stencil
};

export enum class FboType
{
    Planar,
    CubeMap
};

export struct FboAttachment
{
    FboAttachmentType type;
    std::optional<unsigned int> gpuResourceId{};
    unsigned int width;
    unsigned int height;
    TextureFormat captureFormat;
    TextureFormat internalFormat;
    std::vector<unsigned char> data{};
};

export struct Fbo
{
    FboType type;
    std::optional<unsigned int> gpuResourceId{};
    std::vector<Resource<FboAttachment>> attachments;
};

export struct PostProcess {
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
