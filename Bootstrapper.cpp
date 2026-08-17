#include <glad/gl.h>

#include "Bootstrapper.h"

#include <cstdlib>
#include <cstring>
#include <vector>
#include <stb_image_write.h>

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
    renderer.setViewport(window.state().windowWidth, window.state().windowHeight);

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

    dumpFrameIfRequested();
}

void Bootstrapper::dumpFrameIfRequested()
{
    static const char* dumpPath = std::getenv("RACEENGINE_DUMP_FRAME");
    if (dumpPath == nullptr)
    {
        return;
    }

    static int dumpFrameCount = 0;
    if (++dumpFrameCount < 120)
    {
        return;
    }

    const auto& windowState = window.state();
    const auto width = windowState.windowWidth;
    const auto height = windowState.windowHeight;
    const auto rowBytes = static_cast<size_t>(width) * 4;

    auto pixels = std::vector<unsigned char>(rowBytes * static_cast<size_t>(height));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    auto row = std::vector<unsigned char>(rowBytes);
    for (auto y = 0; y < height / 2; y++)
    {
        auto* top = pixels.data() + static_cast<size_t>(y) * rowBytes;
        auto* bottom = pixels.data() + static_cast<size_t>(height - 1 - y) * rowBytes;
        std::memcpy(row.data(), top, rowBytes);
        std::memcpy(top, bottom, rowBytes);
        std::memcpy(bottom, row.data(), rowBytes);
    }

    if (stbi_write_png(dumpPath, width, height, 4, pixels.data(), static_cast<int>(rowBytes)) == 0)
    {
        logger.error("Failed to write frame dump to {}", dumpPath);
        std::exit(1);
    }

    logger.info("Frame dump written to {}", dumpPath);
    std::exit(0);
}