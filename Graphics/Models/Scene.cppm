module;

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/containers/vector.h>
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/maths/soa_transform.h>
#include <ozz/base/maths/vec_float.h>
#include <ozz/options/options.h>

export module raceengine.graphics.models:Scene;

import raceengine.resource;
import :Fbo;
import :Mesh;
import :Shader;
import :Texture;

namespace raceengine
{

export struct SceneNode
{
    SceneNode* parent = nullptr;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::mat4 rotationMatrix = glm::mat4(1.0f);
    glm::mat4 translationMatrix = glm::mat4(1.0f);
    glm::mat4 scaleMatrix = glm::mat4(1.0f);
    glm::vec4 rotation = glm::vec4(0.0f);
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
    glm::vec3 forward = glm::vec3(0.0f);
    // World-transform cache: transformVersion counts recomputes of modelMatrix (0 = never computed);
    // parentTransformVersion is the parent's version this cache was built against (0 = no parent).
    bool transformDirty = true;
    unsigned int transformVersion = 0;
    unsigned int parentTransformVersion = 0;
};

// What the light's geometry is. A point light radiates from `position` and attenuates; a
// directional light has no position that matters and casts along `direction` from infinitely far
// away, which is the light a cascaded shadow map is a shadow map *of*. `direction` is what the
// cascades are fitted against; shading still reads `position` as the direction *towards* the
// light, exactly as it always has, so the two must be opposites of each other for a shadow to land
// where the shading says it should.
export enum class LightType { Point, Directional };

export struct Light
{
    LightType type = LightType::Point;
    glm::vec3 position{};
    glm::vec3 direction{};
    // The four terms both backends upload per light. They are stored as authored rather than
    // derived from color/strength: the renderers hand them to the shaders verbatim, and a
    // product would not reproduce an authored value bit for bit.
    glm::vec3 diffuse{};
    glm::vec3 specular{};
    glm::vec3 ambient{};
    float attenuation = 1.0f;
    glm::vec3 color{};
    float strength{};
};

// How a camera turns view space into clip space. Perspective is the default and the only one any
// camera in a scene has ever used; orthographic is what a directional light's cascade needs,
// because a slice of a directional light's frustum is a box, not a pyramid.
export enum class CameraProjection { Perspective, Orthographic };

// The orthographic view volume's side planes, in view space. Held as four edges rather than a
// width and a height because a cascade's box is fitted to a frustum slice and is not centred on
// the light's axis in general. The near and far planes are Camera's own clipping planes, so an
// orthographic camera and a perspective one state their depth range the same way.
export struct OrthographicVolume
{
    float left = -1.0f;
    float right = 1.0f;
    float bottom = -1.0f;
    float top = 1.0f;
};

// What the frame does with a camera's view. Scene is the ordinary case and the default. A
// ShadowCascade camera *produces* a depth map that Scene cameras sample, so the frame records
// every one of them before any Scene camera whatever order they were appended to the scene in — a
// depth target sampled by a view recorded earlier in the same command buffer would be read before
// it was written, and on Vulkan the layout barrier would be on the wrong side of the read.
export enum class CameraRole { Scene, ShadowCascade };

export struct Camera
{
    unsigned int iso;
    float aspectRatio;
    float aperture;
    float exposure{};
    float fieldOfView;
    CameraRole role = CameraRole::Scene;
    // The shader every draw in this view uses instead of the entity's own. A cascade renders
    // depth and nothing else, and depth does not vary by material, so one shader serves the whole
    // pass; a camera without one shades each entity as that entity asked to be shaded.
    std::optional<Resource<Shader>> overrideShader{};
    CameraProjection projection = CameraProjection::Perspective;
    // Read only when `projection` is Orthographic; `fieldOfView` and `aspectRatio` are read only
    // when it is Perspective. Both sets are kept rather than unioned so switching a camera back
    // and forth does not lose the other mode's framing.
    OrthographicVolume orthographicVolume{};
    // Whether a window resize rebuilds this camera's render target at the new size. True for a
    // camera that fills the screen — every camera built by createCamera() with no argument.
    // False for one that owns a target of its own resolution: a 2048x2048 shadow cascade does not
    // become 1920x1080 because the window did.
    bool tracksWindowSize = true;
    float nearClippingPlane{};
    float farClippingPlane{};
    glm::vec3 position{};
    glm::vec3 direction;
    glm::vec3 roll;
    glm::mat4 modelViewProjectionMatrix{};
    glm::mat4 modelViewMatrix{};
    std::optional<Resource<Fbo>> output;
    std::vector<Resource<PostProcess>> postProcesses{};
};

export enum class RenderableEntityType {
    Mesh,
    Skybox,
};

export struct RenderableEntity
{
    RenderableEntityType type;
    SceneNode& node;
    // Called by the backend immediately around this entity's submission, once per view that
    // draws it. The hooks live on the renderable rather than on the game's Drawable component
    // because this module is the one both sides already depend on: a renderer that had to see
    // a Drawable would make raceengine.graphics import raceengine.game and invert the module
    // graph. Drawable holds them as a view, so a game still writes them through its component.
    std::optional<std::function<void()>> beforeDraw;
    std::optional<std::function<void()>> afterDraw;
    // Whether a shadow cascade's depth pass draws this entity. False for anything that is scenery
    // rather than geometry: a skybox drawn into a cascade fills the whole map at the near plane
    // and puts the entire world in shadow, which is the first thing a shadow map does wrong.
    bool castsShadow = true;

    explicit RenderableEntity(RenderableEntityType type, SceneNode& node) :
        type(type),
        node(node)
    {
    }
};

export struct RenderableMesh
{
    float animationTime{};
    unsigned int currentAnimationIndex{};
    const Resource<Mesh> mesh;
    ozz::vector<ozz::math::SoaTransform> animationLocalSpaceTransforms{};
    ozz::vector<ozz::math::Float4x4> animationModelSpaceTransforms{};
    std::optional<Resource<std::unique_ptr<ozz::animation::Skeleton>>> skeleton;
    std::vector<Resource<std::unique_ptr<ozz::animation::Animation>>> animations{};
    std::unique_ptr<ozz::animation::SamplingJob::Context> animationCache{};
    // ozz joint index -> glTF skin joint index, validated against both when the skeleton is set
    // so the per-frame palette build needs no lookup guards.
    std::map<int, int> jointMap{};
    // Joint palette, rebuilt in place every frame by RenderableEntityService::joints so the draw
    // path does not allocate one vector per mesh per frame.
    std::vector<glm::mat4> jointTransforms{};
};

export struct RenderableModel : public RenderableEntity
{
    Resource<Model> model;
    // The shader this instance was created with. Materials live in shared storage, so the
    // choice cannot be written through to them without one instance rewriting another's.
    Resource<Shader> shader;
    std::vector<RenderableMesh> meshes;

    explicit RenderableModel(SceneNode& node, Resource<Model> model, Resource<Shader> shader,
                             std::vector<RenderableMesh> meshes) :
        RenderableEntity(RenderableEntityType::Mesh, node),
        model(model),
        shader(shader),
        meshes(std::move(meshes))
    {
    }
};

// One slice of the shadow-casting light's frustum, and the camera that draws it.
//
// The camera is a raw pointer into `Scene::cameras` for the same reason `RenderableEntity::node`
// is a reference into `Scene::nodes`: the scene's deques are add-only and the scene is the unit of
// teardown, so an address inside one is valid exactly as long as the scene is. No generational
// handle reaches inside a scene.
export struct ShadowCascade
{
    Camera* camera = nullptr;
    // Where this cascade stops, as a distance along the view camera's axis. The fragment shader
    // picks its cascade by comparing against these.
    float splitDistance = 0.0f;
    // World units per shadow-map texel, and normalised depth per world unit along the light axis.
    // The whole bias budget is expressed in these two, so no cascade needs tuning of its own.
    float texelWorldSize = 1.0f;
    float depthPerWorldUnit = 1.0f;
};

// A scene's cascaded shadow map. Empty `cascades` means the scene casts no shadow, which is a
// legitimate scene rather than a failure — both backends then shade with the lit result.
export struct ShadowCascades
{
    // The light the cascades follow and the camera whose frustum they split. Pointers into the
    // scene's own deques, on the same terms as ShadowCascade::camera.
    Light* light = nullptr;
    Camera* viewCamera = nullptr;
    // Which light index the shaders should apply the shadow to — the position of `light` in
    // Scene::lights, which is the order both backends upload them in.
    unsigned int lightIndex = 0;
    unsigned int resolution = 0;
    // How far from the eye the cascades reach. Beyond it a fragment is lit: there is no map to
    // ask, and the alternative — treating "outside the map" as shadowed — paints the far half of
    // the world black.
    float distance = 0.0f;
    // The split scheme's blend, and how far behind a slice a cascade still catches casters. Kept
    // here rather than only in the DTO because the refit runs every frame and needs both.
    float lambda = 0.5f;
    float casterExtent = 0.0f;
    std::vector<ShadowCascade> cascades{};
};

export struct Scene
{
    // std::deque, and add-only *within a scene*: SceneNode::parent, RenderableEntity::node and a
    // game's own pointer into models are raw references into these containers, and no generational
    // handle reaches inside a scene. The scene is therefore the unit of teardown — destroying it
    // invalidates all of them at once, which is a rule a game can hold, where erasing one node out
    // of the middle is not. There is deliberately no API for the latter.
    std::deque<Camera> cameras;
    std::deque<Light> lights;
    std::deque<RenderableModel> models;
    std::deque<SceneNode> nodes;

    std::optional<Resource<CubeMap>> environment{};
    ShadowCascades shadows{};
};

} // namespace raceengine
