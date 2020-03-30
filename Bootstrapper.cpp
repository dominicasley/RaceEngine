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
    SceneManager& sceneManager) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    resourceService(resourceService),
    window(window),
    renderer(renderer),
    backgroundWorkerService(backgroundWorkerService),
    sceneManager(sceneManager)
{
    renderer.init();

    window.windowResize.subscribe([&, this](auto size) {
        const auto& [width, height] = size;
        logger.info("Window Resized: {}px x {}px", width, height);
        renderer.setViewport(width, height);

        for (auto& camera : sceneManager.getScene("game")->getCameras())
        {
            camera->setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
        }
    });
}

void Bootstrapper::step()
{
    backgroundWorkerService.step();
}

void Bootstrapper::draw()
{
    renderer.draw(sceneManager.getScene("game"));
    window.swapBuffers();
}