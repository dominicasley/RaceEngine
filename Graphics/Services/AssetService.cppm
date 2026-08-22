module;

#include <vector>

#include <spdlog/logger.h>

export module raceengine.graphics:AssetService;

import :IGpuResourceFactory;
import :SceneManagerService;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

// The other half of loading. Releasing a resource is two things that have to happen in this
// order: the GPU objects the element holds ids for go back to the backend, and then the element
// goes out of storage — which is what retires every handle to it. Doing it the other way round
// would lose the ids, which is exactly why nothing could be freed before: the ids lived in the
// model types and the ownership lived in renderer side tables, with no path between them.
//
// Sharing is not modelled. A model owns its meshes, its buffers, its materials and the textures
// those materials name; a cube map owns its six faces. There is no reference count, so unloading
// something two owners named would take it from both — which is why nothing dedupes by path yet
// (see the report accompanying this change).
export class AssetService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    IGpuResourceFactory& gpuResourceFactory;
    SceneManagerService& sceneManagerService;

public:
    explicit AssetService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                          IGpuResourceFactory& gpuResourceFactory, SceneManagerService& sceneManagerService);

    // Releases the model's mesh bindings, vertex/index buffers, material bindings and material
    // textures, then removes every element it owns. Afterwards the model handle, and every mesh,
    // material and texture handle it issued, fails exists(). A renderable still holding the model
    // handle draws nothing rather than reading a recycled slot.
    void unloadModel(const Resource<Model>& modelKey) const;
    void unloadTexture(const Resource<Texture>& textureKey) const;
    // Releases the cube map image and the six face textures it emptied at creation.
    void unloadCubeMap(const Resource<CubeMap>& cubeMapKey) const;
    void unloadShader(const Resource<Shader>& shaderKey) const;
    void unloadFbo(const Resource<Fbo>& fboKey) const;
    void unloadPostProcess(const Resource<PostProcess>& postProcessKey) const;
    // Releases what the scene's cameras own — their render targets and post-process chains — and
    // then destroys the scene. Models are deliberately not touched: they are assets the scene
    // referenced, not state it owned, and the next scene may want them.
    void unloadScene(Scene& scene) const;

    [[nodiscard]] GpuResourceCensus gpuResourceCensus() const;
};

} // namespace raceengine
