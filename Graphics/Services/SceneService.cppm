export module raceengine.graphics:SceneService;

import :RenderableEntityService;
import :CameraService;
import :SceneManagerService;
import raceengine.graphics.models;

namespace raceengine
{

export class SceneService
{
private:
    RenderableEntityService& renderableEntityService;
    CameraService& cameraService;
    SceneManagerService& sceneManagerService;

public:
    explicit SceneService(RenderableEntityService& renderableEntityService, CameraService& cameraService,
                          SceneManagerService& sceneManagerService);
    void update(Scene& scene, float delta) const;
    [[nodiscard]] RenderableModel& createEntity(Scene& scene, const CreateRenderableModelDTO& entityDescriptor) const;
    [[nodiscard]] Camera& createCamera(Scene& scene) const;
    [[nodiscard]] Light& createLight(Scene& scene) const;
    [[nodiscard]] RenderableModel& getModel(Scene& scene, unsigned int index) const;
    [[nodiscard]] Camera& getCamera(Scene& scene, unsigned int index) const;
};

SceneService::SceneService(RenderableEntityService& renderableEntityService, CameraService& cameraService,
                           SceneManagerService& sceneManagerService) :
    renderableEntityService(renderableEntityService),
    cameraService(cameraService),
    sceneManagerService(sceneManagerService)
{
}

// The scene's share of a simulation tick: settle the scene graph once every writer for this
// tick has run. Resolving here rather than lazily from the draw loop means world transforms
// are computed once per tick instead of once per node per camera, and it means the next tick's
// behaviours read positions that already include the previous tick's writes — including a
// parent's, which a child cannot see until the parent's version has been recomputed.
// modelMatrix is idempotent, so a node nothing touched costs a flag test.
//
// delta is unnamed because the scene owns no integrator of its own yet: the one time-varying
// piece of scene state, animation playback, is advanced inside the renderers' draw path
// (recorded in docs/architecture-review.md as a defect — it advances once per camera per
// frame). Moving it here is the fix, and it belongs with that change, not this one.
void SceneService::update(Scene& scene, float) const
{
    for (auto& node : scene.nodes)
    {
        static_cast<void>(sceneManagerService.modelMatrix(node));
    }
}

RenderableModel& SceneService::createEntity(Scene& scene, const CreateRenderableModelDTO& entityDescriptor) const
{
    return scene.models.emplace_back(renderableEntityService.createModel(entityDescriptor));
}

Camera& SceneService::createCamera(Scene& scene) const
{
    return scene.cameras.emplace_back(cameraService.createCamera());
}

Light& SceneService::createLight(Scene& scene) const
{
    return scene.lights.emplace_back(Light());
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
