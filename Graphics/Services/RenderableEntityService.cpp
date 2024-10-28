#include "RenderableEntityService.h"

RenderableEntityService::RenderableEntityService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, AnimationService& animationService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    animationService(animationService)
{

}

RenderableModel RenderableEntityService::createModel(const CreateRenderableModelDTO& entityDescriptor) const
{
    auto createMeshes = [&](const CreateRenderableModelDTO& entityDescriptor) {
        std::vector<RenderableMesh> meshes;

        for (auto materialKey : entityDescriptor.model->materials) {
            auto material = memoryStorageService.materials.get(materialKey);
            material.shader = entityDescriptor.shader;

            memoryStorageService.materials.update(materialKey, material);
        }

        for (const auto& meshKey : entityDescriptor.model->meshes) {
            meshes.emplace_back(RenderableMesh {
                .mesh = meshKey,
                .skeleton = std::nullopt
            });
        }

        return meshes;
    };

    return RenderableModel(entityDescriptor.node, entityDescriptor.model, createMeshes(entityDescriptor));
}

void RenderableEntityService::setAnimation(RenderableMesh& mesh, const std::string& animationName) const
{

}

void RenderableEntityService::setAnimation(RenderableMesh& mesh, unsigned int animationIndex) const
{
    if (mesh.animations.size() > animationIndex)
    {
        mesh.currentAnimationIndex = animationIndex;
        const auto animation = mesh.animations[animationIndex];
    }
}

glm::mat4 ozzToMat4(const ozz::math::Float4x4& t)
{
    return glm::mat4({
     t.cols[0].m128_f32[0], t.cols[0].m128_f32[1], t.cols[0].m128_f32[2], t.cols[0].m128_f32[3],
     t.cols[1].m128_f32[0], t.cols[1].m128_f32[1], t.cols[1].m128_f32[2], t.cols[1].m128_f32[3],
     t.cols[2].m128_f32[0], t.cols[2].m128_f32[1], t.cols[2].m128_f32[2], t.cols[2].m128_f32[3],
     t.cols[3].m128_f32[0], t.cols[3].m128_f32[1], t.cols[3].m128_f32[2], t.cols[3].m128_f32[3]
    });
}

std::vector<glm::mat4>
RenderableEntityService::joints(RenderableMesh& renderableMesh, float frameTimeDelta) const
{
    auto out = std::vector<glm::mat4>();

    if (!renderableMesh.animations.empty())
    {
        auto mesh = renderableMesh.mesh;
        auto animation = renderableMesh.animations[renderableMesh.currentAnimationIndex].value;
        auto skeleton = renderableMesh.skeleton.value().value;

        renderableMesh.animationTime = fmod(renderableMesh.animationTime + frameTimeDelta, animation->get()->duration());

        ozz::animation::SamplingJob sampling_job;
        sampling_job.animation = animation->get();
        sampling_job.cache = renderableMesh.animationCache.get();
        sampling_job.ratio = renderableMesh.animationTime / animation->get()->duration();
        sampling_job.output = ozz::make_span(renderableMesh.animationLocalSpaceTransforms);
        sampling_job.Run();

        ozz::animation::LocalToModelJob ltm_job;
        ltm_job.skeleton = skeleton->get();
        ltm_job.input = ozz::make_span(renderableMesh.animationLocalSpaceTransforms);
        ltm_job.output = ozz::make_span(renderableMesh.animationModelSpaceTransforms);
        ltm_job.Run();

        out.resize(renderableMesh.animationModelSpaceTransforms.size());
        for (auto i = 0; i < renderableMesh.animationModelSpaceTransforms.size(); i++)
        {
            out[renderableMesh.jointMap[i]] =
                ozzToMat4(renderableMesh.animationModelSpaceTransforms[i]) *
                    mesh->inverseBindPoseTransforms[renderableMesh.jointMap[i]];
        }
    }

    return out;
}

void RenderableEntityService::setSkeleton(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Skeleton>> skeletonKey) const
{
    auto skeleton = skeletonKey.value;

    mesh.skeleton = skeletonKey;
    mesh.animationLocalSpaceTransforms.resize(skeleton->get()->num_soa_joints());
    mesh.animationModelSpaceTransforms.resize(skeleton->get()->num_joints());
    mesh.animationCache = std::make_unique<ozz::animation::SamplingCache>(skeleton->get()->num_joints());

    for (auto i = 0; i < skeleton->get()->num_joints(); i++)
    {
        auto m = memoryStorageService.meshes.get(mesh.mesh);
        auto ozzJointName = skeleton->get()->joint_names()[i];

        mesh.jointMap[i] = m.skin.at(ozzJointName);
    }
}

void RenderableEntityService::addAnimation(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Animation>> animation) const
{
    mesh.animations.push_back(animation);
}
