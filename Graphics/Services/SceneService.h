#pragma once

#include <spdlog/logger.h>
#include "../Models/Scene/RenderableEntity.h"
#include "../Models/Scene/Scene.h"
#include "RenderableEntityService.h"
#include "CameraService.h"

class SceneService
{
private:
    spdlog::logger& logger;
    RenderableEntityService& renderableEntityService;
    CameraService& cameraService;

public:
    explicit SceneService(
        spdlog::logger& logger,
        RenderableEntityService& renderableEntityService,
        CameraService& cameraService);
    void update(Scene* scene, float delta) const;
    [[nodiscard]] RenderableEntity* createEntity(Scene* scene, const RenderableEntityDesc& entityDescriptor) const;
    [[nodiscard]] Camera* createCamera(Scene* scene) const;
    [[nodiscard]] Light* createLight(Scene* scene) const;
    [[nodiscard]] RenderableEntity* getEntity(const Scene* scene, unsigned int index) const;
    [[nodiscard]] Camera* getCamera(const Scene* scene, unsigned int index) const;
};