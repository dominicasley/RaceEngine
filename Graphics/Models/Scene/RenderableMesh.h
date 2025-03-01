#pragma once

#include <memory>
#include <vector>
#include <map>
#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/maths/soa_transform.h>
#include <ozz/base/maths/vec_float.h>
#include <ozz/options/options.h>
#include <ozz/base/containers/vector.h>

#include "Mesh.h"

struct RenderableMesh {
    float animationTime;
    unsigned int currentAnimationIndex;
    const Resource<Mesh> mesh;
    ozz::vector<ozz::math::SoaTransform> animationLocalSpaceTransforms;
    ozz::vector<ozz::math::Float4x4> animationModelSpaceTransforms;
    std::optional<Resource<std::unique_ptr<ozz::animation::Skeleton>>> skeleton;
    std::vector<Resource<std::unique_ptr<ozz::animation::Animation>>> animations;
    std::unique_ptr<ozz::animation::SamplingCache> animationCache;
    std::map<int, int> jointMap;
};
