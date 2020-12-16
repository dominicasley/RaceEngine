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

void SceneService::update(Scene* scene, float delta) const
{

}

RenderableModel* SceneService::createEntity(Scene* scene, const CreateRenderableModelDTO& entityDescriptor) const
{
    return scene->models.emplace_back(
        std::make_unique<RenderableModel>(renderableEntityService.createModel(entityDescriptor))
    ).get();
}

RenderableSkybox* SceneService::createEntity(Scene* scene, const CreateRenderableSkyboxDTO& entityDescriptor) const
{
    return scene->skybox.emplace_back(
        std::make_unique<RenderableSkybox>(renderableEntityService.createSkybox(entityDescriptor))
    ).get();
}

Camera* SceneService::createCamera(Scene* scene) const
{
    return scene->cameras.emplace_back(
        std::make_unique<Camera>(cameraService.createCamera())
    ).get();
}

Light* SceneService::createLight(Scene* scene) const
{
    return nullptr;
}

Camera* SceneService::getCamera(const Scene* scene, unsigned int index) const
{
    return scene->cameras[index].get();
}

RenderableModel* SceneService::getModel(const Scene* scene, unsigned int index) const
{
    return scene->models[index].get();;
}


