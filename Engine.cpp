#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "Engine.h"

Engine::Engine() : app(di::make_injector(
    di::bind<spdlog::logger>
        .to(spdlog::stdout_color_mt<spdlog::async_factory>("engine")),
    di::bind<IWindow>
        .in(di::singleton)
        .to<GLFWWindow>(),
    di::bind<VulkanRenderer>
        .in(di::singleton)
        .to<VulkanRenderer>(),
    di::bind<OpenGLRenderer>
        .in(di::singleton)
        .to<OpenGLRenderer>(),
    di::bind<BackgroundWorkerService>
        .in(di::singleton)
        .to<BackgroundWorkerService>(),
    di::bind<MemoryStorageService>
        .in(di::singleton)
        .to<MemoryStorageService>(),
    di::bind<ResourceService>
        .to<ResourceService>(),
    di::bind<GLTFService>
        .to<GLTFService>(),
    di::bind<RenderableEntityService>
        .to<RenderableEntityService>(),
    di::bind<AnimationService>
        .to<AnimationService>(),
    di::bind<SceneService>
        .to<SceneService>(),
    di::bind<ShaderService>
        .to<ShaderService>(),
    di::bind<CubeMapService>
        .to<CubeMapService>(),
    di::bind<FboService>
        .to<FboService>(),
    di::bind<PostProcessService>
        .to<PostProcessService>(),
    di::bind<PresenterService>
        .to<PresenterService>(),
    di::bind<SceneManagerService>
        .in(di::singleton)
        .to<SceneManagerService>())
        .create<Bootstrapper>()),
                   window(app.window),
                   renderer(app.renderer),
                   resource(app.resourceService),
                   memoryStorage(app.memoryStorageService),
                   backgroundWorker(app.backgroundWorkerService),
                   sceneManager(app.sceneManager),
                   scene(app.sceneService),
                   entity(app.renderableEntityService),
                   camera(app.cameraService),
                   shader(app.shaderService),
                   cubeMap(app.cubeMapService),
                   fbo(app.fboService),
                   postProcess(app.postProcessService),
                   presenter(app.presenterService)
{

}

bool Engine::running() const
{
    return !app.window.shouldClose();
}

void Engine::step()
{
    app.step(window.delta());
    app.draw(window.delta());
}
