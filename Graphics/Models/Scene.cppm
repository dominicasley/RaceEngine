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
// away, which is the light a cascaded shadow map is a shadow map *of*. Nothing reads this yet:
// both backends still upload position and the four terms below exactly as before, so declaring a
// light directional changes no shading until the maths that reads it lands.
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

export struct Camera
{
    unsigned int iso;
    float aspectRatio;
    float aperture;
    float exposure{};
    float fieldOfView;
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
};

} // namespace raceengine
