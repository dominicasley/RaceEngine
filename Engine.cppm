module;

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#include <spdlog/async.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

export module raceengine;

export import raceengine.resource;
export import raceengine.graphics.models;
export import raceengine.shared;
export import raceengine.async;
export import raceengine.game;
export import raceengine.graphics;
export import raceengine.io;

// Sandbox code names these unqualified in the global namespace; export import alone
// only surfaces raceengine::X, so re-alias them globally.
export using raceengine::awaitAll;
export using raceengine::Camera;
export using raceengine::CreateRenderableModelDTO;
export using raceengine::Drawable;
export using raceengine::Entity;
export using raceengine::FboAttachmentType;
export using raceengine::Key;
export using raceengine::Presenter;
export using raceengine::RenderableModel;
export using raceengine::Scene;
export using raceengine::SceneNode;
export using raceengine::ShaderDescriptor;

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
    // unique_ptr for backend selection, but the declaration position is still load-bearing:
    // the GL renderer's destructor issues GL deletes that need the GLFW context current, so
    // this member must keep destroying before glfwWindow — same slot the value member had.
    std::unique_ptr<IRenderer> renderer;
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

namespace
{

// Renderer backend selection for the composition root, driven by RACEENGINE_RENDERER.
// Unset or "opengl" selects OpenGL; "vulkan" is accepted but not yet implemented.
[[nodiscard]] std::unique_ptr<IRenderer> createRenderer(spdlog::logger& logger,
                                                        RenderableEntityService& renderableEntityService,
                                                        SceneManagerService& sceneManagerService,
                                                        MemoryStorageService& memoryStorageService)
{
    const char* requested = std::getenv("RACEENGINE_RENDERER");
    if (requested != nullptr && std::string_view(requested) == "vulkan")
    {
        logger.warn("Vulkan renderer requested but not yet available; falling back to OpenGL");
    }

    return std::make_unique<OpenGLRenderer>(logger, renderableEntityService, sceneManagerService, memoryStorageService);
}

} // namespace

Engine::Engine() :
    logger(spdlog::stdout_color_mt<spdlog::async_factory>("engine")),
    glfwWindow(*logger),
    gltfService(*logger, memoryStorageService),
    resourceService(*logger, memoryStorageService, backgroundWorkerService, gltfService),
    renderableEntityService(*logger, memoryStorageService),
    renderer(createRenderer(*logger, renderableEntityService, sceneManagerService, memoryStorageService)),
    fboService(memoryStorageService, *renderer),
    shaderService(memoryStorageService, *renderer),
    cubeMapService(*renderer, memoryStorageService),
    postProcessService(memoryStorageService, fboService, glfwWindow),
    presenterService(*renderer),
    cameraService(memoryStorageService, fboService, glfwWindow),
    sceneService(renderableEntityService, cameraService)
{
    renderer->init();
    renderer->setViewport(glfwWindow.state().windowWidth, glfwWindow.state().windowHeight);

    glfwWindow.onResize(
        [&](int width, int height)
        {
            logger->info("Window Resized: {}px x {}px", width, height);
            renderer->setViewport(width, height);

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
            renderer->draw(scene, camera, delta);
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

    renderer->captureFrame(dumpPath);
    std::exit(0);
}

} // namespace raceengine
