module;

#include <optional>
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
    // See CameraRole and Camera::overrideShader.
    CameraRole role = CameraRole::Scene;
    std::optional<Resource<Shader>> overrideShader{};
};

// What a scene's cascaded shadow map is made of. The defaults are the ones a scene of this size
// wants; a game states the two that depend on its world — how far it wants shadows to reach, and
// how much geometry can stand between the light and the frustum.
export struct CreateShadowCascadesDTO
{
    // The shader every cascade's depth pass draws with. Shaders are the game's assets, so the
    // engine takes the pair rather than owning one: a depth-only vertex stage through the light's
    // matrix and a fragment stage that writes nothing.
    Resource<Shader> depthShader;
    // Square, and unrelated to the window: a cascade target does not resize with it.
    unsigned int resolution = 2048;
    // The practical split scheme's blend. 0 is uniform, 1 is logarithmic. See
    // cascadeSplitDistances.
    float lambda = 0.5f;
    // How far the cascades reach. Fragments beyond it are lit.
    float distance = 2000.0f;
    // How far behind a slice the cascade still catches casters. Roughly the tallest thing in the
    // world, measured along the light.
    float casterExtent = 1500.0f;
};

export struct CreateRenderableModelDTO
{
    SceneNode& node;
    Resource<Shader> shader;
    Resource<Model> model;
};

} // namespace raceengine
