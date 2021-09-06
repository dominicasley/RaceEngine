#pragma once

#include <spdlog/logger.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include "AnimationService.h"
#include <Services/MemoryStorageService.h>
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
    MemoryStorageService& memoryStorageService;
    AnimationService& animationService;

public:
    explicit RenderableEntityService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, AnimationService& animationService);

    [[nodiscard]] RenderableModel createModel(const CreateRenderableModelDTO& entityDescriptor) const;
    [[nodiscard]] RenderableSkybox createSkybox(const CreateRenderableSkyboxDTO& entityDescriptor) const;
    void setSkeleton(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Skeleton>> skeleton) const;
    void addAnimation(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Animation>> animation) const;
    void setAnimation(RenderableMesh& mesh, const std::string& animationName) const;
    void setAnimation(RenderableMesh& mesh, unsigned int animationIndex) const;
    [[nodiscard]] std::vector<glm::mat4> joints(RenderableMesh& renderableMesh, float frameTimeDelta) const;
};
