module;

#include <cmath>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/span.h>

export module raceengine.graphics:RenderableEntityService;

import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class RenderableEntityService
{
private:
    MemoryStorageService& memoryStorageService;

public:
    explicit RenderableEntityService(MemoryStorageService& memoryStorageService);

    [[nodiscard]] RenderableModel createModel(const CreateRenderableModelDTO& entityDescriptor) const;
    [[nodiscard]] std::expected<void, std::string>
    setSkeleton(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Skeleton>> skeleton) const;
    [[nodiscard]] std::expected<void, std::string>
    addAnimation(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Animation>> animation) const;
    [[nodiscard]] std::expected<void, std::string> setAnimation(RenderableMesh& mesh,
                                                                const std::string& animationName) const;
    [[nodiscard]] std::expected<void, std::string> setAnimation(RenderableMesh& mesh,
                                                                unsigned int animationIndex) const;
    // The palette lives on the mesh and is rebuilt in place; it stays valid until this mesh's
    // next joints() call.
    [[nodiscard]] const std::vector<glm::mat4>& joints(RenderableMesh& renderableMesh, float frameTimeDelta) const;
};

RenderableEntityService::RenderableEntityService(MemoryStorageService& memoryStorageService) :
    memoryStorageService(memoryStorageService)
{
}

RenderableModel RenderableEntityService::createModel(const CreateRenderableModelDTO& entityDescriptor) const
{
    // The shader is a per-instance choice but materials are shared storage owned by the Model,
    // so it is written there only as the default for materials that have none: two renderables
    // built from one Resource<Model> would otherwise rewrite each other's materials, the second
    // silently restyling the first. The instance's own choice travels on the RenderableModel.
    for (const auto& materialKey : entityDescriptor.model->materials)
    {
        auto material = memoryStorageService.materials.get(materialKey);
        if (material.shader.has_value())
        {
            continue;
        }

        material.shader = entityDescriptor.shader;
        memoryStorageService.materials.update(materialKey, material);
    }

    auto meshes = std::vector<RenderableMesh>();
    meshes.reserve(entityDescriptor.model->meshes.size());

    for (const auto& meshKey : entityDescriptor.model->meshes)
    {
        meshes.emplace_back(RenderableMesh{.mesh = meshKey, .skeleton = std::nullopt});
    }

    return RenderableModel(entityDescriptor.node, entityDescriptor.model, entityDescriptor.shader, std::move(meshes));
}

std::expected<void, std::string>
RenderableEntityService::setSkeleton(RenderableMesh& mesh,
                                     Resource<std::unique_ptr<ozz::animation::Skeleton>> skeletonKey) const
{
    const auto* skeleton = skeletonKey.value->get();
    const auto& storedMesh = memoryStorageService.meshes.get(mesh.mesh);
    const auto jointCount = static_cast<size_t>(skeleton->num_joints());

    // Resolved and checked before anything is written to the mesh: a skeleton whose joints the
    // glTF skin does not name is a content error, and reporting it here leaves the mesh
    // untouched — drawable but unanimated — rather than half-configured.
    auto resolved = std::vector<int>();
    auto unmapped = std::vector<std::string>();
    resolved.reserve(jointCount);

    for (size_t i = 0; i < jointCount; i++)
    {
        const auto* jointName = skeleton->joint_names()[i];
        const auto skinJoint = storedMesh.skin.find(jointName);

        if (skinJoint == storedMesh.skin.end())
        {
            unmapped.emplace_back(jointName);
            continue;
        }

        if (skinJoint->second < 0 || std::cmp_greater_equal(skinJoint->second, jointCount) ||
            std::cmp_greater_equal(skinJoint->second, storedMesh.inverseBindPoseTransforms.size()))
        {
            return std::unexpected("Joint " + std::string(jointName) + " of mesh " + storedMesh.name +
                                   " maps to skin index " + std::to_string(skinJoint->second) + ", outside the " +
                                   std::to_string(jointCount) + " skeleton joints and " +
                                   std::to_string(storedMesh.inverseBindPoseTransforms.size()) +
                                   " inverse bind matrices");
        }

        resolved.push_back(skinJoint->second);
    }

    if (!unmapped.empty())
    {
        auto names = std::string();
        for (size_t i = 0; i < unmapped.size() && i < 8; i++)
        {
            names += (i == 0 ? "" : ", ") + unmapped[i];
        }

        return std::unexpected("Skeleton and mesh " + storedMesh.name +
                               " disagree: " + std::to_string(unmapped.size()) + " of " + std::to_string(jointCount) +
                               " skeleton joints are absent from the glTF skin (" + names +
                               (unmapped.size() > 8 ? ", ..." : "") + ")");
    }

    mesh.skeleton = skeletonKey;
    mesh.jointMap.clear();
    for (size_t i = 0; i < resolved.size(); i++)
    {
        mesh.jointMap[static_cast<int>(i)] = resolved[i];
    }

    mesh.animationLocalSpaceTransforms.resize(static_cast<size_t>(skeleton->num_soa_joints()));
    mesh.animationModelSpaceTransforms.resize(jointCount);
    mesh.animationCache = std::make_unique<ozz::animation::SamplingJob::Context>(skeleton->num_joints());
    mesh.jointTransforms.clear();

    return {};
}

std::expected<void, std::string>
RenderableEntityService::addAnimation(RenderableMesh& mesh,
                                      Resource<std::unique_ptr<ozz::animation::Animation>> animationKey) const
{
    const auto* animation = animationKey.value->get();

    // Sampling reads into buffers sized from the skeleton, so the skeleton has to be set
    // first; without this the mismatch only shows up as a throw from inside the draw loop.
    if (!mesh.skeleton.has_value())
    {
        return std::unexpected("Cannot add animation " + std::string(animation->name()) + " to mesh " +
                               memoryStorageService.meshes.get(mesh.mesh).name + ": it has no skeleton yet");
    }

    // A clip with more tracks than the skeleton has joints fails ozz's own job validation,
    // which would otherwise surface as a mesh that simply never moves.
    const auto* skeleton = mesh.skeleton.value().value->get();
    if (animation->num_tracks() > skeleton->num_joints())
    {
        return std::unexpected("Animation " + std::string(animation->name()) + " drives " +
                               std::to_string(animation->num_tracks()) + " tracks but the skeleton of mesh " +
                               memoryStorageService.meshes.get(mesh.mesh).name + " has " +
                               std::to_string(skeleton->num_joints()) + " joints");
    }

    mesh.animations.push_back(animationKey);

    return {};
}

std::expected<void, std::string> RenderableEntityService::setAnimation(RenderableMesh& mesh,
                                                                       const std::string& animationName) const
{
    // ozz keeps the clip name in the archive, so the name lookup needs no side table; the
    // selection itself is the index overload's job.
    for (size_t i = 0; i < mesh.animations.size(); i++)
    {
        if (animationName == mesh.animations[i].value->get()->name())
        {
            return setAnimation(mesh, static_cast<unsigned int>(i));
        }
    }

    return std::unexpected("No animation named " + animationName + " on mesh " +
                           memoryStorageService.meshes.get(mesh.mesh).name);
}

std::expected<void, std::string> RenderableEntityService::setAnimation(RenderableMesh& mesh,
                                                                       unsigned int animationIndex) const
{
    if (animationIndex >= mesh.animations.size())
    {
        return std::unexpected("Animation index " + std::to_string(animationIndex) + " is out of range for mesh " +
                               memoryStorageService.meshes.get(mesh.mesh).name + ", which carries " +
                               std::to_string(mesh.animations.size()) + " animation(s)");
    }

    mesh.currentAnimationIndex = animationIndex;

    return {};
}

glm::mat4 ozzToMat4(const ozz::math::Float4x4& t)
{
    // Both types are 16 contiguous floats in column-major order.
    glm::mat4 out;
    std::memcpy(&out, &t.cols, sizeof(out));
    return out;
}

const std::vector<glm::mat4>& RenderableEntityService::joints(RenderableMesh& renderableMesh,
                                                              float frameTimeDelta) const
{
    // Rebuilt in place. clear() + resize() value-initializes every entry exactly as the
    // freshly allocated vector this used to return did, while keeping the buffer's capacity,
    // so an animated mesh stops allocating a palette per frame.
    auto& out = renderableMesh.jointTransforms;
    out.clear();

    // addAnimation refuses a mesh with no skeleton, so this is the other half of that
    // contract rather than a condition to report from inside the draw loop.
    if (renderableMesh.animations.empty() || !renderableMesh.skeleton.has_value())
    {
        return out;
    }

    const auto& mesh = renderableMesh.mesh;
    const auto* animation = renderableMesh.animations[renderableMesh.currentAnimationIndex].value->get();
    const auto* skeleton = renderableMesh.skeleton.value().value->get();

    renderableMesh.animationTime = std::fmod(renderableMesh.animationTime + frameTimeDelta, animation->duration());

    ozz::animation::SamplingJob samplingJob;
    samplingJob.animation = animation;
    samplingJob.context = renderableMesh.animationCache.get();
    samplingJob.ratio = renderableMesh.animationTime / animation->duration();
    samplingJob.output = ozz::make_span(renderableMesh.animationLocalSpaceTransforms);

    ozz::animation::LocalToModelJob localToModelJob;
    localToModelJob.skeleton = skeleton;
    localToModelJob.input = ozz::make_span(renderableMesh.animationLocalSpaceTransforms);
    localToModelJob.output = ozz::make_span(renderableMesh.animationModelSpaceTransforms);

    if (!samplingJob.Run() || !localToModelJob.Run())
    {
        return out;
    }

    out.resize(renderableMesh.animationModelSpaceTransforms.size());

    // jointMap is dense over the skeleton's joints and both its keys and its values were
    // range-checked in setSkeleton, so the palette build needs no lookup or bounds guards.
    for (const auto& [ozzJoint, meshJoint] : renderableMesh.jointMap)
    {
        out[static_cast<size_t>(meshJoint)] =
            ozzToMat4(renderableMesh.animationModelSpaceTransforms[static_cast<size_t>(ozzJoint)]) *
            mesh->inverseBindPoseTransforms[static_cast<size_t>(meshJoint)];
    }

    return out;
}

} // namespace raceengine
