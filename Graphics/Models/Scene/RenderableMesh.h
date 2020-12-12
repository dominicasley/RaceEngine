#pragma once

#include <memory>
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
    const Mesh* mesh;
    unsigned int currentAnimationIndex;
    ozz::animation::Skeleton* skeleton;
    ozz::vector<ozz::math::SoaTransform> animationLocalSpaceTransforms;
    ozz::vector<ozz::math::Float4x4> animationModelSpaceTransforms;
    std::vector<ozz::animation::Animation*> animations;
    std::unique_ptr<ozz::animation::SamplingCache> animationCache;
    std::map<int, int> jointMap;
};
