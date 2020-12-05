#include "RenderableEntityService.h"

RenderableEntityService::RenderableEntityService(spdlog::logger& logger, AnimationService& animationService) :
    logger(logger),
    animationService(animationService)
{

}

RenderableEntity RenderableEntityService::createEntity(const RenderableEntityDesc& entityDescriptor) const
{
    return RenderableEntity{
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

void RenderableEntityService::setPosition(RenderableEntity& entity, float x, float y, float z) const
{
    entity.position = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
}

void RenderableEntityService::setDirection(RenderableEntity& entity, float angle, float x, float y, float z) const
{
    entity.rotation = glm::rotate(glm::mat4(1.0), angle, glm::vec3(x, y, z));
}

void RenderableEntityService::setScale(RenderableEntity& entity, float x, float y, float z) const
{
    entity.scale = glm::scale(glm::mat4(1.0f), glm::vec3(x, y, z));
}

void RenderableEntityService::translate(RenderableEntity& entity, float x, float y, float z) const
{
    entity.position = glm::translate(entity.position, glm::vec3(x, y, z));
}

void RenderableEntityService::rotate(RenderableEntity& entity, float angle, float x, float y, float z) const
{
    entity.rotation = glm::rotate(entity.rotation, angle, glm::vec3(x, y, z));
}

void RenderableEntityService::scale(RenderableEntity& entity, float x, float y, float z) const
{
    entity.scale = glm::scale(entity.scale, glm::vec3(x, y, z));
}

void RenderableEntityService::setParent(RenderableEntity& entity, RenderableEntity* parent) const
{
    entity.parent = parent;
}

void RenderableEntityService::setAnimation(RenderableEntity& entity, const std::string& animationName) const
{
    const auto animation = std::find_if(
        entity.model->animations.begin(),
        entity.model->animations.end(),
        [&animationName](const auto& animation) {
            return animation.name == animationName;
        });

    if (animation != entity.model->animations.end())
    {
        logger.info(
            "set animation: {} {} samplers {} channels",
            animation->name,
            animation->samplers.size(),
            animation->channels.size());

        entity.currentAnimationIndex = static_cast<unsigned int>(animation - entity.model->animations.begin());
    }
}

void RenderableEntityService::setAnimation(RenderableEntity& entity, unsigned int animationIndex) const
{
    if (entity.model->animations.size() > animationIndex)
    {
        entity.currentAnimationIndex = animationIndex;

        const auto animation = entity.model->animations[animationIndex];

        logger.info(
            "set animation: {} {} samplers {} channels",
            animation.name,
            animation.samplers.size(),
            animation.channels.size());

        for (const auto& channel : animation.channels)
        {
            const auto keyFrames = AccessorUtility::get<std::vector<float>>(*entity.model,
                                                                            entity.model->accessors[animation.samplers[channel.sampler].input]);

            if (channel.target_path == "translation")
            {
                const auto positions = AccessorUtility::get<std::vector<glm::vec3>>(
                    *entity.model, entity.model->accessors[animation.samplers[channel.sampler].output]);

                for (auto i = 0; i < keyFrames.size(); i++)
                {
                    entity.position_state[channel.target_node][keyFrames[i]] = positions[i];
                }
            }
            else if (channel.target_path == "rotation")
            {
                const auto rotations = AccessorUtility::get<std::vector<glm::quat>>(
                    *entity.model, entity.model->accessors[animation.samplers[channel.sampler].output]);

                for (auto i = 0; i < keyFrames.size(); i++)
                {
                    entity.rotation_state[channel.target_node][keyFrames[i]] = rotations[i];
                }
            }
            else if (channel.target_path == "scale")
            {
                const auto scale = AccessorUtility::get<std::vector<glm::vec3>>(
                    *entity.model, entity.model->accessors[animation.samplers[channel.sampler].output]);

                for (auto i = 0; i < keyFrames.size(); i++)
                {
                    entity.scaling_state[channel.target_node][keyFrames[i]] = scale[i];
                }
            }
        }
    }
}

glm::mat4
animate(std::map<int, glm::mat4>& at, RenderableEntity& entity, float x, const tinygltf::Model* model,
        const tinygltf::Node& node, const int& i, glm::mat4 t)
{
    if (at.find(i) != at.end())
    {
        return at[i];
    }

    auto nextPosition = glm::vec3(0.0f);
    auto nextRotation = glm::toQuat(glm::mat4(1.0f));
    auto nextScale = glm::vec3(0.0f);

    if (entity.position_state[i].size() > 0)
    {
        const auto m = fmod(x, entity.position_state[i].rbegin()->first);
        const auto currentPosition = entity.position_state[i].lower_bound(m)->second;
        nextPosition = entity.position_state[i].upper_bound(m)->second;
        // const auto percentT = (m - currentPosition->first) / (nextPosition->first - currentPosition->first);
    }

    if (entity.rotation_state[i].size() > 0)
    {
        const auto n = fmod(x, entity.rotation_state[i].rbegin()->first);
        const auto currentRotation = entity.rotation_state[i].lower_bound(n);
        nextRotation = entity.rotation_state[i].upper_bound(n)->second;
        // const auto percentR = (n - currentRotation->first) / (nextRotation->first - currentRotation->first);
    }

    if (entity.scaling_state[i].size() > 0)
    {
        const auto o = fmod(x, entity.scaling_state[i].rbegin()->first);
        const auto currentScale = entity.scaling_state[i].lower_bound(o);
        nextScale = entity.scaling_state[i].upper_bound(o)->second;
        // const auto percentS = (o - currentScale->first) / (nextScale->first - currentScale->first);
    }


    t *= (glm::translate(glm::mat4(1.0f), nextPosition) *
          glm::toMat4(nextRotation) *
          glm::scale(glm::mat4(1.0f), nextScale));

    at[i] = t;

    for (const auto& ci : node.children)
    {
        animate(at, entity, x, model, model->nodes[ci], ci, t);
    }

    return at[i];
}

std::vector<glm::mat4>
RenderableEntityService::joints(RenderableEntity& entity, const tinygltf::Model* model, tinygltf::Node& node) const
{
    static float x = 0.0f;
    x += 0.001f;

    return std::vector<glm::mat4>();
}

const glm::mat4& RenderableEntityService::modelMatrix(RenderableEntity& entity) const
{
    entity.modelMatrix = entity.position * entity.rotation * entity.scale;

    if (entity.parent)
    {
        entity.modelMatrix = modelMatrix(*entity.parent) * entity.modelMatrix;
    }

    entity.forward = normalize(glm::vec3(glm::inverse(entity.modelMatrix)[2]));

    return entity.modelMatrix;
}

