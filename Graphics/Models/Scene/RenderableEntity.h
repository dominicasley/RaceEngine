#pragma once

#include <memory>
#include <map>
#include <optional>
#include <glm/glm.hpp>
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

#include "../../Api/Resources/RenderableEntityDesc.h"

struct RenderableEntity
{
    RenderableEntity* parent;
    tinygltf::Model* model;
    ozz::animation::Skeleton* skeleton;
    ozz::vector<ozz::math::SoaTransform> animationLocalSpaceTransforms;
    ozz::vector<ozz::math::Float4x4> animationModelSpaceTransforms;
    std::vector<glm::mat4> inverseBindPoseTransforms;
    std::vector<ozz::animation::Animation*> animations;
    glm::mat4 modelMatrix;
    glm::mat4 rotation;
    glm::mat4 position;
    glm::mat4 scale;
    glm::vec3 forward;
    unsigned int currentAnimationIndex;
    std::unique_ptr<ozz::animation::SamplingCache> animationCache;
};



