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

RenderableEntity* SceneService::createEntity(Scene& scene, const RenderableEntityDesc& entityDescriptor) const
{
    return scene.entities.emplace_back(
        std::make_unique<RenderableEntity>(renderableEntityService.createEntity(entityDescriptor))
    ).get();
}

Camera* SceneService::createCamera(Scene& scene) const
{
    return scene.cameras.emplace_back(
        std::make_unique<Camera>(cameraService.createCamera())
    ).get();
}

Light* SceneService::createLight(Scene& scene) const
{
    return nullptr;
}

Camera* SceneService::getCamera(const Scene& scene, unsigned int index) const
{
    return scene.cameras[index].get();
}

RenderableEntity* SceneService::getEntity(const Scene& scene, unsigned int index) const
{
    return scene.entities[index].get();;
}


