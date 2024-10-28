#include "Bootstrapper.h"

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
    CubeMapService& cubeMapService,
    FboService& fboService,
    PostProcessService& postProcessService,
    PresenterService& presenterService,
    EntityService& entityService) :
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
    cubeMapService(cubeMapService),
    fboService(fboService),
    postProcessService(postProcessService),
    presenterService(presenterService),
    entityService(entityService)
{
    renderer.init();

    window.windowResize.subscribe([&](auto size) {
        const auto& [width, height] = size;
        logger.info("Window Resized: {}px x {}px", width, height);
        renderer.setViewport(width, height);

        for (auto& scene : sceneManager.getScenes())
        {
            for (auto& camera : scene.cameras)
            {
                cameraService.setAspectRatio(camera, static_cast<float>(width) / static_cast<float>(height));
                cameraService.recreateOutputBuffer(camera, width, height);

                for (auto postProcess : camera.postProcesses)
                {
                    postProcessService.recreateOutputBuffer(postProcess, width, height);
                }
            }
        }
    });
}

void Bootstrapper::step(float delta)
{
    backgroundWorkerService.step();

    for (auto& scene : sceneManager.getScenes())
    {
        for (auto& camera : scene.cameras)
        {
            cameraService.updateModelViewProjectionMatrix(camera);
        }
    }
}

void Bootstrapper::draw(float delta)
{
    for (auto& scene : sceneManager.getScenes())
    {
        for (auto& camera : scene.cameras)
        {
            renderer.draw(scene, camera, delta);
        }
    }

    presenterService.present();
    window.swapBuffers();
}