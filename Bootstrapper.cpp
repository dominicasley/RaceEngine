//
// Created by Dominic on 6/11/2019.
//

#include "Bootstrapper.h"

#include <utility>

Bootstrapper::Bootstrapper(
    spdlog::logger& logger,
    MemoryStorageService& memoryStorageService,
    ResourceService& resourceService,
    IWindow& window,
    OpenGLRenderer& renderer,
    BackgroundWorkerService& backgroundWorkerService,
    SceneManagerService& sceneManager,
    SceneService& sceneService,
    RenderableEntityService& renderableEntityService,
    CameraService& cameraService,
    ShaderService& shaderService,
    CubeMapService& cubeMapService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    resourceService(resourceService),
    window(window),
    renderer(renderer),
    backgroundWorkerService(backgroundWorkerService),
    sceneManager(sceneManager),
    sceneService(sceneService),
    renderableEntityService(renderableEntityService),
    cameraService(cameraService),
    shaderService(shaderService),
    cubeMapService(cubeMapService)
{
    renderer.init();

    window.windowResize.subscribe([&, this](auto size) {
        const auto& [width, height] = size;
        logger.info("Window Resized: {}px x {}px", width, height);
        renderer.setViewport(width, height);

        for (auto& camera : sceneManager.getScene("game")->cameras)
        {
            cameraService.setAspectRatio(camera.get(), static_cast<float>(width) / static_cast<float>(height));
        }
    });
}

void Bootstrapper::step(float delta)
{
    backgroundWorkerService.step();
}

void Bootstrapper::draw(float delta)
{
    for (auto& scene : sceneManager.getScenes())
    {
        renderer.draw(scene.second.get(), delta);
    }

    window.swapBuffers();
}