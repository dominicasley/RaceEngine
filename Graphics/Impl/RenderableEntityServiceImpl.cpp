// RenderableEntityService bodies. Declarations are in Graphics/Services/RenderableEntityService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
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

module raceengine.graphics;

import :RenderableEntityService;
import :FrameDiagnostics;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

std::string RenderableEntityService::meshName(const RenderableMesh& mesh) const
{
    const auto* stored = memoryStorageService.meshes.find(mesh.mesh);

    return stored == nullptr ? "<unloaded mesh " + std::to_string(mesh.mesh.index) + ">" : stored->name;
}

RenderableEntityService::RenderableEntityService(MemoryStorageService& memoryStorageService,
                                                 FrameDiagnostics& diagnostics) :
    memoryStorageService(memoryStorageService),
    diagnostics(diagnostics)
{
}

RenderableModel RenderableEntityService::createModel(const CreateRenderableModelDTO& entityDescriptor) const
{
    // Resolved once and read through: the model is the handle's element, and every list below
    // lives inside it. A model whose handle went stale between the load and here builds an
    // entity with no meshes, which the draw path already skips.
    const auto* model = memoryStorageService.models.find(entityDescriptor.model);
    if (model == nullptr)
    {
        return RenderableModel(entityDescriptor.node, entityDescriptor.model, entityDescriptor.shader,
                               std::vector<RenderableMesh>());
    }

    // The shader is a per-instance choice but materials are shared storage owned by the Model,
    // so it is written there only as the default for materials that have none: two renderables
    // built from one Resource<Model> would otherwise rewrite each other's materials, the second
    // silently restyling the first. The instance's own choice travels on the RenderableModel.
    for (const auto& materialKey : model->materials)
    {
        memoryStorageService.materials.mutate(materialKey,
                                              [&](Material& material)
                                              {
                                                  if (!material.shader.has_value())
                                                  {
                                                      material.shader = entityDescriptor.shader;
                                                  }
                                              });
    }

    auto meshes = std::vector<RenderableMesh>();
    meshes.reserve(model->meshes.size());

    for (const auto& meshKey : model->meshes)
    {
        meshes.emplace_back(RenderableMesh{.mesh = meshKey, .skeleton = std::nullopt});
    }

    return RenderableModel(entityDescriptor.node, entityDescriptor.model, entityDescriptor.shader, std::move(meshes));
}

std::expected<void, std::string>
RenderableEntityService::setSkeleton(RenderableMesh& mesh,
                                     Resource<std::unique_ptr<ozz::animation::Skeleton>> skeletonKey) const
{
    const auto* skeletonSlot = memoryStorageService.skeletons.find(skeletonKey);
    const auto* storedMeshSlot = memoryStorageService.meshes.find(mesh.mesh);
    if (skeletonSlot == nullptr || storedMeshSlot == nullptr)
    {
        return std::unexpected("the skeleton or the mesh it would be set on is no longer loaded");
    }

    const auto* skeleton = skeletonSlot->get();
    const auto& storedMesh = *storedMeshSlot;
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
    const auto* animationSlot = memoryStorageService.animations.find(animationKey);
    if (animationSlot == nullptr)
    {
        return std::unexpected("the animation is no longer loaded");
    }

    const auto* animation = animationSlot->get();

    // Sampling reads into buffers sized from the skeleton, so the skeleton has to be set
    // first; without this the mismatch only shows up as a throw from inside the draw loop.
    if (!mesh.skeleton.has_value())
    {
        return std::unexpected("Cannot add animation " + std::string(animation->name()) + " to mesh " + meshName(mesh) +
                               ": it has no skeleton yet");
    }

    const auto* skeletonSlot = memoryStorageService.skeletons.find(mesh.skeleton.value());
    if (skeletonSlot == nullptr)
    {
        return std::unexpected("Cannot add animation " + std::string(animation->name()) + " to mesh " + meshName(mesh) +
                               ": its skeleton is no longer loaded");
    }

    // A clip with more tracks than the skeleton has joints fails ozz's own job validation,
    // which would otherwise surface as a mesh that simply never moves.
    const auto* skeleton = skeletonSlot->get();
    if (animation->num_tracks() > skeleton->num_joints())
    {
        return std::unexpected("Animation " + std::string(animation->name()) + " drives " +
                               std::to_string(animation->num_tracks()) + " tracks but the skeleton of mesh " +
                               meshName(mesh) + " has " + std::to_string(skeleton->num_joints()) + " joints");
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
        const auto* animation = memoryStorageService.animations.find(mesh.animations[i]);
        if (animation != nullptr && animationName == (*animation)->name())
        {
            return setAnimation(mesh, static_cast<unsigned int>(i));
        }
    }

    return std::unexpected("No animation named " + animationName + " on mesh " + meshName(mesh));
}

std::expected<void, std::string> RenderableEntityService::setAnimation(RenderableMesh& mesh,
                                                                       unsigned int animationIndex) const
{
    if (animationIndex >= mesh.animations.size())
    {
        return std::unexpected("Animation index " + std::to_string(animationIndex) + " is out of range for mesh " +
                               meshName(mesh) + ", which carries " + std::to_string(mesh.animations.size()) +
                               " animation(s)");
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

    // Three lookups per animated mesh per frame, all of them past the early return above, so an
    // unanimated mesh pays none of them and an animated one pays them once rather than once per
    // joint. Any of the three going stale means the clip cannot be sampled, which is the same
    // answer as having no clip: an empty palette, and the draw records an unanimated mesh.
    const auto* mesh = memoryStorageService.meshes.find(renderableMesh.mesh);
    const auto* animationSlot =
        memoryStorageService.animations.find(renderableMesh.animations[renderableMesh.currentAnimationIndex]);
    const auto* skeletonSlot = memoryStorageService.skeletons.find(renderableMesh.skeleton.value());
    if (mesh == nullptr || animationSlot == nullptr || skeletonSlot == nullptr)
    {
        return out;
    }

    const auto* animation = animationSlot->get();
    const auto* skeleton = skeletonSlot->get();

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

    // ozz validates its jobs rather than reporting from inside them, so a refusal here means the
    // clip and the skeleton disagree about something setSkeleton/addAnimation did not catch. The
    // mesh renders unanimated either way; what changes is that the reason is no longer lost.
    if (!samplingJob.Run() || !localToModelJob.Run())
    {
        diagnostics.record(FrameDiagnostic::SkinningRejected,
                           [&] { return "mesh " + mesh->name + ", clip " + std::string(animation->name()); });

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
