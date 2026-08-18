module;

#include <memory>

#include <spdlog/logger.h>

// The one unit in raceengine.graphics that imports both backends, and an implementation
// partition so that importing it is not something a consumer of the module can do. Everything
// Vulkan, VMA, shaderc, glad and GLFW bring with them stops here.
module raceengine.graphics:RenderBackendFactoryImpl;

import :FrameDiagnostics;
import :GraphicsApi;
import :IRenderBackend;
import :OpenGLRenderer;
import :RenderBackendFactory;
import :RenderableEntityService;
import :SceneManagerService;
import :VulkanRenderer;
import :Window;
import raceengine.shared;

namespace raceengine
{

std::unique_ptr<IRenderBackend>
createRenderer(GraphicsApi graphicsApi, spdlog::logger& logger, FrameDiagnostics& frameDiagnostics, IWindow& window,
               IVulkanSurfaceSource& surfaceSource, RenderableEntityService& renderableEntityService,
               SceneManagerService& sceneManagerService, MemoryStorageService& memoryStorageService)
{
    if (graphicsApi == GraphicsApi::Vulkan)
    {
        return std::make_unique<VulkanRenderer>(logger, frameDiagnostics, window, surfaceSource,
                                                renderableEntityService, sceneManagerService, memoryStorageService);
    }

    return std::make_unique<OpenGLRenderer>(logger, frameDiagnostics, renderableEntityService, sceneManagerService,
                                            memoryStorageService);
}

} // namespace raceengine
