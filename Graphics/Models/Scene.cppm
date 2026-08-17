module;

#include <deque>
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
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/maths/soa_transform.h>
#include <ozz/base/maths/vec_float.h>
#include <ozz/options/options.h>
#include <ozz/base/containers/vector.h>

export module raceengine.graphics.models:Scene;

import raceengine.resource;
import :Fbo;
import :Mesh;

namespace raceengine
{

export struct SceneNode {
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

export struct Light
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 color;
    float strength;
};

export struct Camera
{
    unsigned int iso;
    float aspectRatio;
    float aperture;
    float exposure;
    float fieldOfView;
    float nearClippingPlane;
    float farClippingPlane;
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 roll;
    glm::mat4 modelViewProjectionMatrix;
    glm::mat4 modelViewMatrix;
    std::optional<Resource<Fbo>> output;
    std::vector<Resource<PostProcess>> postProcesses;
};

export enum class RenderableEntityType {
    Mesh,
    Skybox,
};

export struct RenderableEntity
{
    RenderableEntityType type;
    SceneNode& node;

    explicit RenderableEntity(RenderableEntityType type, SceneNode& node) : type(type), node(node) {}
};

export struct RenderableMesh {
    float animationTime;
    unsigned int currentAnimationIndex;
    const Resource<Mesh> mesh;
    ozz::vector<ozz::math::SoaTransform> animationLocalSpaceTransforms;
    ozz::vector<ozz::math::Float4x4> animationModelSpaceTransforms;
    std::optional<Resource<std::unique_ptr<ozz::animation::Skeleton>>> skeleton;
    std::vector<Resource<std::unique_ptr<ozz::animation::Animation>>> animations;
    std::unique_ptr<ozz::animation::SamplingJob::Context> animationCache;
    std::map<int, int> jointMap;
};

export struct RenderableModel : public RenderableEntity
{
    Resource<Model> model;
    std::vector<RenderableMesh> meshes;

    explicit RenderableModel(SceneNode& node, Resource<Model> model, std::vector<RenderableMesh> meshes) :
        RenderableEntity(RenderableEntityType::Mesh, node),
        model(model),
        meshes(std::move(meshes)) { }
};

export struct Scene
{
    // std::deque: element addresses stay stable under growth; references into these containers rely on it, so no erasing.
    std::deque<Camera> cameras;
    std::deque<Light> lights;
    std::deque<RenderableModel> models;
    std::deque<SceneNode> nodes;
};

} // namespace raceengine
