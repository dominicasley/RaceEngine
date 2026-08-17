module;

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <glad/gl.h>
#include <spdlog/async.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stb_image_write.h>

#include "Graphics/Models/Scene/CreateRenderableModelDTO.h"
#include "Graphics/Models/Scene/FboAttachmentType.h"
#include "Graphics/Models/Scene/Presenter.h"
#include "Graphics/Models/Scene/Scene.h"
#include "Graphics/Models/Scene/ShaderDescriptor.h"
#include "Io/Services/GLTFService.h"
#include "Shared/Shared.h"

export module raceengine;

export import raceengine.async;
export import raceengine.game;
export import raceengine.graphics;
export import raceengine.io;

export using ::Camera;
export using ::CreateRenderableModelDTO;
export using ::FboAttachmentType;
export using ::Presenter;
export using ::RenderableModel;
export using ::Scene;
export using ::SceneNode;
export using ::ShaderDescriptor;

// Sandbox code names these unqualified in the global namespace; export import alone
// only surfaces raceengine::X, so re-alias them globally.
export using raceengine::awaitAll;
export using raceengine::Drawable;
export using raceengine::Entity;

namespace raceengine
{

export class Engine
{
private:
    // Declaration order IS initialization order and reverse-destruction order:
    // each member may only depend on members declared above it.
    std::shared_ptr<spdlog::logger> logger;
    GLFWWindow glfwWindow;
    MemoryStorageService memoryStorageService;
    BackgroundWorkerService backgroundWorkerService;
    GLTFService gltfService;
    ResourceService resourceService;
    RenderableEntityService renderableEntityService;
    SceneManagerService sceneManagerService;
    OpenGLRenderer openGlRenderer;
    FboService fboService;
    ShaderService shaderService;
    CubeMapService cubeMapService;
    PostProcessService postProcessService;
    PresenterService presenterService;
    CameraService cameraService;
    SceneService sceneService;
    EntityService entityService;

public:
    Engine();
    [[nodiscard]] bool running() const;
    void step();

    [[nodiscard]] IWindow& window()
    {
        return glfwWindow;
    }

    [[nodiscard]] ResourceService& resource()
    {
        return resourceService;
    }

    [[nodiscard]] MemoryStorageService& memoryStorage()
    {
        return memoryStorageService;
    }

    [[nodiscard]] SceneManagerService& sceneManager()
    {
        return sceneManagerService;
    }

    [[nodiscard]] SceneService& scene()
    {
        return sceneService;
    }

    [[nodiscard]] RenderableEntityService& renderableEntity()
    {
        return renderableEntityService;
    }

    [[nodiscard]] CameraService& camera()
    {
        return cameraService;
    }

    [[nodiscard]] ShaderService& shader()
    {
        return shaderService;
    }

    [[nodiscard]] CubeMapService& cubeMap()
    {
        return cubeMapService;
    }

    [[nodiscard]] FboService& fbo()
    {
        return fboService;
    }

    [[nodiscard]] PostProcessService& postProcess()
    {
        return postProcessService;
    }

    [[nodiscard]] PresenterService& presenter()
    {
        return presenterService;
    }

    [[nodiscard]] EntityService& entity()
    {
        return entityService;
    }

private:
    void dumpFrameIfRequested();
};

} // namespace raceengine

module :private;

namespace raceengine
{

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

    glfwWindow.onResize([&](int width, int height) {
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

} // namespace raceengine
