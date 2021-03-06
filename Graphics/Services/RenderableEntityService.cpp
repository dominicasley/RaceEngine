#include "RenderableEntityService.h"

RenderableEntityService::RenderableEntityService(spdlog::logger& logger, AnimationService& animationService) :
    logger(logger),
    animationService(animationService)
{

}

RenderableModel RenderableEntityService::createModel(const CreateRenderableModelDTO& entityDescriptor) const
{
    auto createMeshes = [](const CreateRenderableModelDTO& entityDescriptor) {
        std::vector<RenderableMesh> meshes;

        for (const auto& m : entityDescriptor.model->meshes) {
            auto material = !m.materials.empty() ? m.materials.front() : nullptr;

            if (material != nullptr)
            {
                material->shader = entityDescriptor.shader;
            }

            meshes.push_back(RenderableMesh {
                .mesh = &m,
                .material = material,
                .skeleton = nullptr
            });
        }

        return meshes;
    };

    return RenderableModel(entityDescriptor.node, entityDescriptor.model, createMeshes(entityDescriptor));
}

RenderableSkybox RenderableEntityService::createSkybox(const CreateRenderableSkyboxDTO& entityDescriptor) const
{
    auto skybox = RenderableSkybox(
        entityDescriptor.node,
        entityDescriptor.cubeMap,
        entityDescriptor.shader,
        entityDescriptor.model
    );

    return skybox;
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

        logger.info(
            "set animation: {} {} duration {} channels",
            animation->name(),
            animation->duration(),
            animation->num_tracks());
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
RenderableEntityService::joints(RenderableMesh& mesh, float frameTimeDelta) const
{
    auto out = std::vector<glm::mat4>();

    if (!mesh.animations.empty())
    {
        mesh.animationTime = fmod(mesh.animationTime + frameTimeDelta, mesh.animations[mesh.currentAnimationIndex]->duration());

        ozz::animation::SamplingJob sampling_job;
        sampling_job.animation = mesh.animations[mesh.currentAnimationIndex];
        sampling_job.cache = mesh.animationCache.get();
        sampling_job.ratio = mesh.animationTime / mesh.animations[mesh.currentAnimationIndex]->duration();
        sampling_job.output = ozz::make_span(mesh.animationLocalSpaceTransforms);
        sampling_job.Run();

        ozz::animation::LocalToModelJob ltm_job;
        ltm_job.skeleton = mesh.skeleton;
        ltm_job.input = ozz::make_span(mesh.animationLocalSpaceTransforms);
        ltm_job.output = ozz::make_span(mesh.animationModelSpaceTransforms);
        ltm_job.Run();

        out.resize(mesh.animationModelSpaceTransforms.size());
        for (auto i = 0; i < mesh.animationModelSpaceTransforms.size(); i++)
        {
            out[mesh.jointMap[i]] =
                ozzToMat4(mesh.animationModelSpaceTransforms[i]) *
                mesh.mesh->inverseBindPoseTransforms[mesh.jointMap[i]];
        }
    }

    return out;
}

void RenderableEntityService::setSkeleton(RenderableMesh& mesh, ozz::animation::Skeleton* skeleton) const
{
    mesh.skeleton = skeleton;

    mesh.animationLocalSpaceTransforms.resize(skeleton->num_soa_joints());
    mesh.animationModelSpaceTransforms.resize(skeleton->num_joints());
    mesh.animationCache = std::make_unique<ozz::animation::SamplingCache>(skeleton->num_joints());

    for (auto i = 0; i < mesh.skeleton->num_joints(); i++)
    {
        auto ozzJointName = mesh.skeleton->joint_names()[i];
        mesh.jointMap[i] = mesh.mesh->skin.at(ozzJointName);
    }
}

void RenderableEntityService::addAnimation(RenderableMesh& mesh, ozz::animation::Animation* animation) const
{
    mesh.animations.push_back(animation);
}
