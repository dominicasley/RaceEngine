#include "RenderableEntityService.h"

RenderableEntityService::RenderableEntityService(spdlog::logger& logger, AnimationService& animationService) :
    logger(logger),
    animationService(animationService)
{

}

RenderableEntity RenderableEntityService::createEntity(const RenderableEntityDesc& entityDescriptor) const
{
    return RenderableEntity {
        .parent = nullptr,
        .model =  entityDescriptor.model,
        .modelMatrix = glm::mat4(1.0f),
        .rotation = glm::mat4(1.0f),
        .position = glm::mat4(1.0f),
        .scale = glm::mat4(1.0f),
        .forward = glm::vec3(0.0f),
        .currentAnimationIndex = 0,
    };
}

void RenderableEntityService::setPosition(RenderableEntity* entity, float x, float y, float z) const
{
    entity->position = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
}

void RenderableEntityService::setDirection(RenderableEntity* entity, float angle, float x, float y, float z) const
{
    entity->rotation = glm::rotate(glm::mat4(1.0), angle, glm::vec3(x, y, z));
}

void RenderableEntityService::setScale(RenderableEntity* entity, float x, float y, float z) const
{
    entity->scale = glm::scale(glm::mat4(1.0f), glm::vec3(x, y, z));
}

void RenderableEntityService::translate(RenderableEntity* entity, float x, float y, float z) const
{
    entity->position = glm::translate(entity->position, glm::vec3(x, y, z));
}

void RenderableEntityService::rotate(RenderableEntity* entity, float angle, float x, float y, float z) const
{
    entity->rotation = glm::rotate(entity->rotation, angle, glm::vec3(x, y, z));
}

void RenderableEntityService::scale(RenderableEntity* entity, float x, float y, float z) const
{
    entity->scale = glm::scale(entity->scale, glm::vec3(x, y, z));
}

void RenderableEntityService::setParent(RenderableEntity* entity, RenderableEntity* parent) const
{
    entity->parent = parent;
}

void RenderableEntityService::setAnimation(RenderableEntity* entity, const std::string& animationName) const
{

}

void RenderableEntityService::setAnimation(RenderableEntity* entity, unsigned int animationIndex) const
{
    if (entity->animations.size() > animationIndex)
    {
        entity->currentAnimationIndex = animationIndex;

        const auto animation = entity->animations[animationIndex];

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
RenderableEntityService::joints(RenderableEntity* entity) const
{
    auto out = std::vector<glm::mat4>();

    if (!entity->animations.empty())
    {
        static float x = 0.0f;
        x += 0.0001f;
        x = fmod(x, 1.0);

        ozz::animation::SamplingJob sampling_job;
        sampling_job.animation = entity->animations[entity->currentAnimationIndex];
        sampling_job.cache = entity->animationCache.get();
        sampling_job.ratio = x;
        sampling_job.output = ozz::make_span(entity->animationLocalSpaceTransforms);
        sampling_job.Run();

        ozz::animation::LocalToModelJob ltm_job;
        ltm_job.skeleton = entity->skeleton;
        ltm_job.input = ozz::make_span(entity->animationLocalSpaceTransforms);
        ltm_job.output = ozz::make_span(entity->animationModelSpaceTransforms);
        ltm_job.Run();

        std::map<int, int> jointMap;
        for (auto i = 0; i < entity->skeleton->num_joints(); i++)
        {
            auto ozzJointName = entity->skeleton->joint_names()[i];
            jointMap[i] = entity->model->meshes[0].skin[ozzJointName];
        }

        out.resize(entity->animationModelSpaceTransforms.size());
        for (auto i = 0; i < entity->animationModelSpaceTransforms.size(); i++)
        {
            out[jointMap[i]] =
                ozzToMat4(entity->animationModelSpaceTransforms[i]) *
                entity->model->meshes[0].inverseBindPoseTransforms[jointMap[i]];
        }
    }

    return out;
}

const glm::mat4& RenderableEntityService::modelMatrix(RenderableEntity* entity) const
{
    entity->modelMatrix = entity->position * entity->rotation * entity->scale;

    if (entity->parent)
    {
        entity->modelMatrix = modelMatrix(entity->parent) * entity->modelMatrix;
    }

    entity->forward = normalize(glm::vec3(glm::inverse(entity->modelMatrix)[2]));

    return entity->modelMatrix;
}

void RenderableEntityService::setSkeleton(RenderableEntity* entity, ozz::animation::Skeleton* skeleton) const
{
    entity->skeleton = skeleton;

    entity->animationLocalSpaceTransforms.resize(skeleton->num_soa_joints());
    entity->animationModelSpaceTransforms.resize(skeleton->num_joints());
    entity->animationCache = std::make_unique<ozz::animation::SamplingCache>(skeleton->num_joints());
}

void RenderableEntityService::addAnimation(RenderableEntity* entity, ozz::animation::Animation* animation) const
{
    entity->animations.push_back(animation);
}
