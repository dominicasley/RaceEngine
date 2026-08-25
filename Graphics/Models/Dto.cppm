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
    // See FboAttachment::levels. One is a plain render target.
    unsigned int levels = 1;
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
    // Zero is spellable now because the `output` form below does not read them; a camera that owns
    // its target still has to state both, and a zero reaches the backend's minimised-window clamp.
    unsigned int width = 0;
    unsigned int height = 0;
    CameraTarget target = CameraTarget::ColourAndDepth;
    // Applied to the depth attachment, if there is one. See DepthComparison.
    DepthComparison depthComparison = DepthComparison::None;
    // See CameraRole and Camera::overrideShader.
    CameraRole role = CameraRole::Scene;
    std::optional<Resource<Shader>> overrideShader{};
    // A render target this camera draws into instead of owning one — typically an Fbo composed over
    // attachments other framebuffers already own (FboService::compose), which is how a layered
    // frame's cameras share one depth buffer. When set, the width, height, target and
    // depthComparison above are not read: the attachments already answer all four.
    std::optional<Resource<Fbo>> output{};
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

// What a camera's bloom is made of. Two shaders and the dial settings; how long the chain is and
// which level each pass writes are the engine's, because they are a function of the view's size.
export struct CreateBloomDTO
{
    // Halves the frame down the first chain. One shader for every level of it, told which it is in
    // the fullscreen push constant: level zero reads the scene and thresholds it, and the rest read
    // the level above. The same shape as the exposure meter's reduction, and for the same reason.
    Resource<Shader> downsampleShader;
    // Carries the result back up the second chain, combining each level with the matching level of
    // the first. Fullscreen, two inputs.
    Resource<Shader> upsampleShader;
    // Threshold, knee, intensity and spread, defaulted by Bloom itself.
    Bloom bloom{};
};

// What a camera's ambient occlusion is made of: the three shaders it takes and how strong the
// result is. The buffers are the engine's to size — they are the view's own resolution, which the
// camera already knows — and the order the passes run in is not a game's decision either.
export struct CreateAmbientOcclusionDTO
{
    // Draws the view's geometry into the prepass buffer: view-space normal in rgb, view depth in a.
    // A pair of its own rather than the entities' shaders because what a prepass writes does not
    // vary by material, exactly as a shadow cascade's does not.
    Resource<Shader> prepassShader;
    // Gathers the horizon around each pixel of that buffer. Fullscreen.
    Resource<Shader> gatherShader;
    // Smooths the gathered term along surfaces rather than across them, reading the prepass buffer
    // again to tell the two apart. Fullscreen, and separate from the gather because a blur that ran
    // inside it would be sampling a buffer it was still writing.
    Resource<Shader> blurShader;
    // Strength and radius, defaulted by AmbientOcclusion itself; the state fields are overwritten
    // when it is enabled.
    AmbientOcclusion occlusion{};
};

// What a camera's exposure meter is made of. One shader and the dial settings; the chain it reduces
// through is the engine's to size and to build, because its length has to agree with the level the
// backend copies back and neither is a game's business.
export struct CreateAutoExposureDTO
{
    // The fullscreen shader every level of the reduction is drawn with. One shader, not two: a pass
    // is told which level it is writing in the fullscreen push constant, and the only level that
    // differs is the first, which reads scene colour where the rest read a partial reduction.
    Resource<Shader> shader;
    // The dial settings and adaptation rates, defaulted by AutoExposure itself. A level that wants
    // a different compensation writes that field and leaves the rest; the state fields are
    // overwritten when the meter is enabled, so nothing here can seed a reading.
    AutoExposure meter{};
};

export struct CreateRenderableModelDTO
{
    SceneNode& node;
    Resource<Shader> shader;
    Resource<Model> model;
};

} // namespace raceengine
