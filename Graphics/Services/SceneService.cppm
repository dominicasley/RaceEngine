export module raceengine.graphics:SceneService;

import :RenderableEntityService;
import :CameraService;
import raceengine.graphics.models;

namespace raceengine
{

export class SceneService
{
private:
    RenderableEntityService& renderableEntityService;
    CameraService& cameraService;

public:
    explicit SceneService(
        RenderableEntityService& renderableEntityService,
        CameraService& cameraService);
    void update(Scene& scene, float delta) const;
    [[nodiscard]] RenderableModel& createEntity(Scene& scene, const CreateRenderableModelDTO& entityDescriptor) const;
    [[nodiscard]] Camera& createCamera(Scene& scene) const;
    [[nodiscard]] Light& createLight(Scene& scene) const;
    [[nodiscard]] RenderableModel& getModel(Scene& scene, unsigned int index) const;
    [[nodiscard]] Camera& getCamera(Scene& scene, unsigned int index) const;
};

SceneService::SceneService(
    RenderableEntityService& renderableEntityService,
    CameraService& cameraService) :
    renderableEntityService(renderableEntityService),
    cameraService(cameraService)
{

}

void SceneService::update(Scene&, float) const
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

} // namespace raceengine
