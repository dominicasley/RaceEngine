export module raceengine.graphics;

export import :Window;
export import :GraphicsApi;
export import :RenderContract;
export import :ShadowCascades;
export import :FrameDiagnostics;
export import :IFrameRecorder;
export import :IGpuResourceFactory;
export import :IFrameCapture;
export import :IRenderBackend;
// The concrete backends are deliberately absent: :VulkanRenderer and :OpenGLRenderer are
// implementation partitions, reachable only through createRenderer below. Exporting them put
// vulkan/vulkan.h, vk_mem_alloc.h, shaderc, glad and GLFW into every importer's closure.
export import :RenderBackendFactory;
export import :SceneManagerService;
export import :RenderableEntityService;
export import :FboService;
export import :ShaderService;
export import :CubeMapService;
export import :PresenterService;
export import :CameraService;
export import :PostProcessService;
export import :SceneService;
export import :ShadowService;
export import :AssetService;
