module;

#include <memory>

#include <spdlog/logger.h>

export module raceengine.graphics:RenderBackendFactory;

import :FrameDiagnostics;
import :IRenderBackend;
import :RenderableEntityService;
import :SceneManagerService;
import :Window;
import raceengine.shared;

namespace raceengine
{

// The only public way to obtain a backend, and the reason the concrete backend is an
// implementation partition. This unit names IRenderBackend and nothing below it, so importing
// raceengine.graphics costs a consumer the seams and not Vulkan, VMA, shaderc or GLFW.
//
// The definition lives in :RenderBackendFactoryImpl — the one unit in the module that imports
// the backend — because a definition here would put it back in the interface closure this
// declaration exists to keep it out of.
//
// The backend takes the storage service to write GPU ids back through their Resources, and the
// scene services the draw path reads node transforms and joint matrices from.
//
// `window` and `surfaceSource` are the same object seen through two seams. It is a separate
// parameter rather than a wider IWindow because IWindow is what a game is handed.
export [[nodiscard]] std::unique_ptr<IRenderBackend>
createRenderer(spdlog::logger& logger, FrameDiagnostics& frameDiagnostics, IWindow& window,
               IVulkanSurfaceSource& surfaceSource, RenderableEntityService& renderableEntityService,
               SceneManagerService& sceneManagerService, MemoryStorageService& memoryStorageService);

} // namespace raceengine
