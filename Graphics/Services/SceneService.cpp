#include "SceneService.h"

SceneService::SceneService(
    spdlog::logger& logger,
    RenderableEntityService& renderableEntityService,
    CameraService& cameraService) :
    logger(logger),
    renderableEntityService(renderableEntityService),
    cameraService(cameraService)
{

}

void SceneService::update(Scene& scene, float delta) const
{

}

RenderableModel& SceneService::createEntity(Scene& scene, const CreateRenderableModelDTO& entityDescriptor) const
{
    return scene.models.emplace_back(
        renderableEntityService.createModel(entityDescriptor)
    );
}

Camera& SceneService::createCamera(Scene& scene) const
{
    return scene.cameras.emplace_back(
        cameraService.createCamera()
    );
}

Light& SceneService::createLight(Scene& scene) const
{
    return scene.lights.emplace_back(
        Light()
    );
}

Camera& SceneService::getCamera(Scene& scene, unsigned int index) const
{
    return scene.cameras.at(index);
}

RenderableModel& SceneService::getModel(Scene& scene, unsigned int index) const
{
    return scene.models.at(index);
}


