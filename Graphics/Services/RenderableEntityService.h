#pragma once

#include <spdlog/logger.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include "AnimationService.h"
#include "../../Io/Utility/AccessorUtility.h"
#include "../Models/Scene/RenderableEntityDesc.h"
#include "../Models/Scene/RenderableEntity.h"

class RenderableEntityService
{
private:
    spdlog::logger& logger;
    AnimationService& animationService;

public:
    explicit RenderableEntityService(spdlog::logger& logger, AnimationService& animationService);

    [[nodiscard]] RenderableEntity createEntity(const RenderableEntityDesc& entityDescriptor) const;
    void setSkeleton(RenderableMesh& mesh, ozz::animation::Skeleton* skeleton) const;
    void addAnimation(RenderableMesh& mesh, ozz::animation::Animation* animation) const;
    void setAnimation(RenderableMesh& mesh, const std::string& animationName) const;
    void setAnimation(RenderableMesh& mesh, unsigned int animationIndex) const;
    [[nodiscard]] std::vector<glm::mat4> joints(RenderableMesh& entity, float frameTimeDelta) const;
};
