export module raceengine.graphics;

export import :Window;
export import :RenderContract;
export import :SphericalHarmonics;
export import :ShadowCascades;
export import :PostProcessing;
export import :LookupTable;
export import :PhysicalCamera;
export import :FrameDiagnostics;
export import :IFrameRecorder;
export import :IGpuResourceFactory;
export import :IFrameCapture;
export import :IRenderBackend;
// The concrete backend is deliberately absent: :VulkanRenderer is an implementation partition,
// reachable only through createRenderer below. Exporting it put vulkan/vulkan.h, vk_mem_alloc.h,
// shaderc and GLFW into every importer's closure.
export import :RenderBackendFactory;
export import :SceneManagerService;
export import :RenderableEntityService;
export import :FboService;
export import :ShaderService;
export import :CubeMapService;
export import :LightProbeService;
export import :PresenterService;
export import :CameraService;
export import :PostProcessService;
export import :AutoExposureService;
export import :AmbientOcclusionService;
export import :BloomService;
export import :ColourGradeService;
export import :SceneService;
export import :ShadowService;
export import :AssetService;
