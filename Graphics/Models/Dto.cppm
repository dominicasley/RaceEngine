module;

#include <vector>

export module raceengine.graphics.models:Dto;

import raceengine.resource;
import :Fbo;
import :Mesh;
import :Scene;
import :Shader;
import :Texture;

namespace raceengine
{

export struct CreateFboAttachmentDTO
{
    unsigned int width;
    unsigned int height;
    FboAttachmentType type;
    TextureFormat captureFormat;
    TextureFormat internalFormat;
};

export struct CreateFboDTO
{
    FboType type;
    std::vector<CreateFboAttachmentDTO> attachments;
};

export struct CreateRenderableModelDTO
{
    SceneNode& node;
    Resource<Shader> shader;
    Resource<Model> model;
};

} // namespace raceengine
