#pragma once

#include <spdlog/logger.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include "AnimationService.h"
#include "../../Io/Utility/AccessorUtility.h"
#include "../Models/Scene/RenderableEntity.h"

class RenderableEntityService
{
private:
    spdlog::logger& logger;
    AnimationService& animationService;

public:
    explicit RenderableEntityService(spdlog::logger& logger, AnimationService& animationService);

    [[nodiscard]] RenderableEntity createEntity(const RenderableEntityDesc& entityDescriptor) const;
    void setPosition(RenderableEntity* entity, float x, float y, float z) const;
    void setDirection(RenderableEntity* entity, float angle, float x, float y, float z) const;
    void setScale(RenderableEntity* entity, float x, float y, float z) const;

    void translate(RenderableEntity* entity, float x, float y, float z) const;
    void rotate(RenderableEntity* entity, float angle, float x, float y, float z) const;
    void scale(RenderableEntity* entity, float x, float y, float z) const;

    void setParent(RenderableEntity* entity, RenderableEntity* parent) const;
    void setSkeleton(RenderableEntity* entity, ozz::animation::Skeleton* skeleton) const;
    void addAnimation(RenderableEntity* entity, ozz::animation::Animation* animation) const;
    void setAnimation(RenderableEntity* entity, const std::string& animationName) const;
    void setAnimation(RenderableEntity* entity, unsigned int animationIndex) const;

    [[nodiscard]] std::vector<glm::mat4> joints(RenderableEntity* entity) const;
    [[nodiscard]] const glm::mat4& modelMatrix(RenderableEntity* entity) const;
};
