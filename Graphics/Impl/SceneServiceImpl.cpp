// SceneService bodies. Declarations are in Graphics/Services/SceneService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <functional>
#include <string>
#include <utility>

module raceengine.graphics;

import :SceneService;
import :RenderableEntityService;
import :CameraService;
import :SceneManagerService;
import raceengine.graphics.models;

namespace raceengine
{

SceneService::SceneService(RenderableEntityService& renderableEntityService, CameraService& cameraService,
                           SceneManagerService& sceneManagerService) :
    renderableEntityService(renderableEntityService),
    cameraService(cameraService),
    sceneManagerService(sceneManagerService)
{
}

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

std::expected<std::reference_wrapper<Camera>, std::string> SceneService::createCamera(Scene& scene) const
{
    auto camera = cameraService.createCamera();
    if (!camera)
    {
        return std::unexpected(camera.error());
    }

    return std::ref(scene.cameras.emplace_back(std::move(camera).value()));
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
