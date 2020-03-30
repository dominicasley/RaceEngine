#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "Engine.h"

Engine::Engine() :
        app(di::make_injector(
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
                di::bind<SceneManager>
                        .in(di::singleton)
                        .to<SceneManager>()).create<Bootstrapper>()),
        window(app.window),
        renderer(app.renderer),
        resourceService(app.resourceService),
        memoryStorageService(app.memoryStorageService),
        backgroundWorkerService(app.backgroundWorkerService),
        sceneManager(app.sceneManager) {
}

bool Engine::running() const {
    return !app.window.shouldClose();
}

void Engine::step() {
    app.step();
    app.draw();
}
