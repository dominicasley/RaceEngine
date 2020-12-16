#pragma once

#include <spdlog/logger.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include "AnimationService.h"
#include "../../Io/Utility/AccessorUtility.h"
#include "../Models/Scene/CreateRenderableModelDTO.h"
#include "../Models/Scene/CreateRenderableSkyboxDTO.h"
#include "../Models/Scene/RenderableEntity.h"
#include "../Models/Scene/RenderableMesh.h"
#include "../Models/Scene/RenderableModel.h"
#include "../Models/Scene/RenderableSkybox.h"

class RenderableEntityService
{
private:
    spdlog::logger& logger;
    AnimationService& animationService;

public:
    explicit RenderableEntityService(spdlog::logger& logger, AnimationService& animationService);

    [[nodiscard]] RenderableModel createModel(const CreateRenderableModelDTO& entityDescriptor) const;
    [[nodiscard]] RenderableSkybox createSkybox(const CreateRenderableSkyboxDTO& entityDescriptor) const;
    void setSkeleton(RenderableMesh& mesh, ozz::animation::Skeleton* skeleton) const;
    void addAnimation(RenderableMesh& mesh, ozz::animation::Animation* animation) const;
    void setAnimation(RenderableMesh& mesh, const std::string& animationName) const;
    void setAnimation(RenderableMesh& mesh, unsigned int animationIndex) const;
    [[nodiscard]] std::vector<glm::mat4> joints(RenderableMesh& entity, float frameTimeDelta) const;
};
