#include <glad/gl.h>

#include "Engine.h"

#include <cstdlib>
#include <cstring>
#include <vector>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stb_image_write.h>

Engine::Engine() :
    logger(spdlog::stdout_color_mt<spdlog::async_factory>("engine")),
    glfwWindow(*logger),
    memoryStorageService(*logger),
    backgroundWorkerService(*logger),
    gltfService(*logger, memoryStorageService),
    resourceService(*logger, memoryStorageService, backgroundWorkerService, gltfService),
    renderableEntityService(*logger, memoryStorageService),
    sceneManagerService(*logger),
    openGlRenderer(*logger, renderableEntityService, sceneManagerService, memoryStorageService),
    fboService(*logger, memoryStorageService, openGlRenderer),
    shaderService(*logger, memoryStorageService, openGlRenderer),
    cubeMapService(*logger, openGlRenderer, memoryStorageService),
    postProcessService(*logger, memoryStorageService, fboService, glfwWindow),
    presenterService(*logger, openGlRenderer),
    cameraService(*logger, memoryStorageService, fboService, glfwWindow),
    sceneService(*logger, renderableEntityService, cameraService),
    entityService(*logger)
{
    openGlRenderer.init();
    openGlRenderer.setViewport(glfwWindow.state().windowWidth, glfwWindow.state().windowHeight);

    glfwWindow.windowResize.subscribe([&](auto size) {
        const auto& [width, height] = size;
        logger->info("Window Resized: {}px x {}px", width, height);
        openGlRenderer.setViewport(width, height);

        for (auto& scene : sceneManagerService.getScenes())
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

bool Engine::running() const
{
    return !glfwWindow.shouldClose();
}

void Engine::step()
{
    const auto delta = glfwWindow.delta();

    backgroundWorkerService.step();

    for (auto& scene : sceneManagerService.getScenes())
    {
        for (auto& camera : scene.cameras)
        {
            cameraService.updateModelViewProjectionMatrix(camera);
        }
    }

    for (auto& scene : sceneManagerService.getScenes())
    {
        for (auto& camera : scene.cameras)
        {
            openGlRenderer.draw(scene, camera, delta);
        }
    }

    presenterService.present();
    glfwWindow.swapBuffers();

    dumpFrameIfRequested();
}

void Engine::dumpFrameIfRequested()
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

    const auto& windowState = glfwWindow.state();
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
        logger->error("Failed to write frame dump to {}", dumpPath);
        std::exit(1);
    }

    logger->info("Frame dump written to {}", dumpPath);
    std::exit(0);
}
