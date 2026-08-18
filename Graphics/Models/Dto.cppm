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
    DepthComparison depthComparison = DepthComparison::None;
};

export struct CreateFboDTO
{
    FboType type;
    std::vector<CreateFboAttachmentDTO> attachments;
};

// Which attachments a camera's render target carries. Colour and depth is what an on-screen
// camera has always had; depth only is what a shadow cascade or a depth pre-pass draws into, and
// colour only is a reflection probe or post-FX target that needs no depth test. There is no
// fourth case on purpose — a framebuffer with no attachments is not a render target, and making
// it unspellable is cheaper than reporting it.
export enum class CameraTarget { ColourAndDepth, ColourOnly, DepthOnly };

// The render target a camera is built with. `CameraService::createCamera()` with no argument
// still means "colour and depth at the window's size, resized with the window", which is every
// camera that exists today; anything built from this DTO owns a target of its own size that a
// window resize leaves alone.
export struct CreateCameraDTO
{
    unsigned int width;
    unsigned int height;
    CameraTarget target = CameraTarget::ColourAndDepth;
    // Applied to the depth attachment, if there is one. See DepthComparison.
    DepthComparison depthComparison = DepthComparison::None;
};

export struct CreateRenderableModelDTO
{
    SceneNode& node;
    Resource<Shader> shader;
    Resource<Model> model;
};

} // namespace raceengine
