module;

#include <memory>

#include <spdlog/logger.h>

// The one unit in raceengine.graphics that imports the backend, and an implementation partition
// so that importing it is not something a consumer of the module can do. Everything Vulkan, VMA,
// shaderc and GLFW bring with them stops here.
module raceengine.graphics:RenderBackendFactoryImpl;

import :FrameDiagnostics;
import :IRenderBackend;
import :RenderBackendFactory;
import :RenderableEntityService;
import :SceneManagerService;
import :VulkanRenderer;
import :Window;
import raceengine.shared;

namespace raceengine
{

std::unique_ptr<IRenderBackend> createRenderer(spdlog::logger& logger, FrameDiagnostics& frameDiagnostics,
                                               IWindow& window, IVulkanSurfaceSource& surfaceSource,
                                               RenderableEntityService& renderableEntityService,
                                               SceneManagerService& sceneManagerService,
                                               MemoryStorageService& memoryStorageService);

} // namespace raceengine
