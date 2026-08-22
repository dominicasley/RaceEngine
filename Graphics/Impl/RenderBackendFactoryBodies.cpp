// RenderBackendFactoryImpl bodies. Declarations are in Graphics/Api/RenderBackendFactoryImpl.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <memory>

#include <spdlog/logger.h>

// The one unit in raceengine.graphics that imports the backend, and an implementation partition
// so that importing it is not something a consumer of the module can do. Everything Vulkan, VMA,
// shaderc and GLFW bring with them stops here.

module raceengine.graphics;

import :RenderBackendFactoryImpl;
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
                                               MemoryStorageService& memoryStorageService)
{
    return std::make_unique<VulkanRenderer>(logger, frameDiagnostics, window, surfaceSource, renderableEntityService,
                                            sceneManagerService, memoryStorageService);
}

} // namespace raceengine
