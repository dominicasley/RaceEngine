// AssetService bodies. Declarations are in Graphics/Services/AssetService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <vector>

#include <spdlog/logger.h>

module raceengine.graphics;

import :AssetService;
import :IGpuResourceFactory;
import :SceneManagerService;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

AssetService::AssetService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                           IGpuResourceFactory& gpuResourceFactory, SceneManagerService& sceneManagerService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    gpuResourceFactory(gpuResourceFactory),
    sceneManagerService(sceneManagerService)
{
}

void AssetService::unloadTexture(const Resource<Texture>& textureKey) const
{
    const auto* texture = memoryStorageService.textures.find(textureKey);
    if (texture == nullptr)
    {
        // Already released. One glTF image becomes one Texture element that several materials of
        // the same model name, so this arrives once per naming and must free once.
        return;
    }

    if (texture->gpuResourceId.has_value())
    {
        gpuResourceFactory.releaseGpuResource(GpuResourceKind::Texture, texture->gpuResourceId.value());
    }

    memoryStorageService.textures.remove(textureKey);
}

void AssetService::unloadModel(const Resource<Model>& modelKey) const
{
    const auto* model = memoryStorageService.models.find(modelKey);
    if (model == nullptr)
    {
        return;
    }

    const auto before = gpuResourceFactory.gpuResourceCensus();

    // Copied out first: the walk below removes the elements these lists live inside.
    const auto meshKeys = model->meshes;
    const auto materialKeys = model->materials;

    auto bufferIds = std::vector<unsigned int>();
    bufferIds.reserve(model->meshBuffers.size());
    for (const auto& meshBuffer : model->meshBuffers)
    {
        if (meshBuffer.gpuId.has_value())
        {
            bufferIds.push_back(meshBuffer.gpuId.value());
        }
    }

    // Per-primitive bindings before the buffers they name — on Vulkan a binding holds the
    // VkBuffer handles, on GL a VAO holds the element-array binding — so the referrer goes first.
    for (const auto& meshKey : meshKeys)
    {
        if (const auto* mesh = memoryStorageService.meshes.find(meshKey); mesh != nullptr)
        {
            for (const auto& primitive : mesh->meshPrimitives)
            {
                if (primitive.gpuVao.has_value())
                {
                    gpuResourceFactory.releaseGpuResource(GpuResourceKind::VertexArray, primitive.gpuVao.value());
                }
            }
        }

        memoryStorageService.meshes.remove(meshKey);
    }

    for (const auto bufferId : bufferIds)
    {
        gpuResourceFactory.releaseGpuResource(GpuResourceKind::Buffer, bufferId);
    }

    for (const auto& materialKey : materialKeys)
    {
        if (const auto* material = memoryStorageService.materials.find(materialKey); material != nullptr)
        {
            for (const auto& textureKey : {material->albedo, material->metallicRoughness, material->normal,
                                           material->occlusion, material->emissive})
            {
                if (textureKey.has_value())
                {
                    unloadTexture(textureKey.value());
                }
            }

            for (const auto& textureKey : material->textures)
            {
                unloadTexture(textureKey);
            }
        }

        gpuResourceFactory.releaseMaterial(materialKey);
        memoryStorageService.materials.remove(materialKey);
    }

    memoryStorageService.models.remove(modelKey);

    const auto after = gpuResourceFactory.gpuResourceCensus();
    logger.info("Model unloaded: {} mesh(es), {} material(s); GPU objects {} -> {} ({} buffer(s), {} texture(s), {} "
                "mesh binding(s) released)",
                meshKeys.size(), materialKeys.size(), before.total(), after.total(), before.buffers - after.buffers,
                before.textures - after.textures, before.vertexArrays - after.vertexArrays);
}

void AssetService::unloadCubeMap(const Resource<CubeMap>& cubeMapKey) const
{
    const auto* cubeMap = memoryStorageService.cubeMaps.find(cubeMapKey);
    if (cubeMap == nullptr)
    {
        return;
    }

    const auto faceKeys = {cubeMap->front, cubeMap->back, cubeMap->left, cubeMap->right, cubeMap->top, cubeMap->bottom};
    const auto gpuResourceId = cubeMap->gpuResourceId;

    gpuResourceFactory.releaseGpuResource(GpuResourceKind::CubeMap, gpuResourceId);
    memoryStorageService.cubeMaps.remove(cubeMapKey);

    for (const auto& faceKey : faceKeys)
    {
        unloadTexture(faceKey);
    }
}

void AssetService::unloadShader(const Resource<Shader>& shaderKey) const
{
    const auto* shader = memoryStorageService.shaders.find(shaderKey);
    if (shader == nullptr)
    {
        return;
    }

    gpuResourceFactory.releaseGpuResource(GpuResourceKind::ShaderProgram, shader->gpuResourceId);
    memoryStorageService.shaders.remove(shaderKey);
}

void AssetService::unloadFbo(const Resource<Fbo>& fboKey) const
{
    const auto* stored = memoryStorageService.frameBuffers.find(fboKey);
    if (stored == nullptr)
    {
        return;
    }

    // deleteFbo takes the framebuffer by reference and clears the ids it released, including the
    // attachments' — through their own elements — so it is handed a copy and the elements go
    // straight after.
    auto frameBuffer = *stored;
    gpuResourceFactory.deleteFbo(frameBuffer);

    for (const auto& attachmentKey : frameBuffer.attachments)
    {
        memoryStorageService.bufferAttachments.remove(attachmentKey);
    }

    memoryStorageService.frameBuffers.remove(fboKey);
}

void AssetService::unloadPostProcess(const Resource<PostProcess>& postProcessKey) const
{
    const auto* postProcess = memoryStorageService.postProcesses.find(postProcessKey);
    if (postProcess == nullptr)
    {
        return;
    }

    const auto output = postProcess->output;
    memoryStorageService.postProcesses.remove(postProcessKey);

    if (output.has_value())
    {
        unloadFbo(output.value());
    }
}

void AssetService::unloadScene(Scene& scene) const
{
    for (auto& camera : scene.cameras)
    {
        for (const auto& postProcessKey : camera.postProcesses)
        {
            unloadPostProcess(postProcessKey);
        }

        if (camera.output.has_value())
        {
            unloadFbo(camera.output.value());
        }
    }

    const auto cameraCount = scene.cameras.size();
    const auto modelCount = scene.models.size();

    sceneManagerService.destroyScene(scene);

    logger.info("Scene unloaded: {} camera target(s) released, {} renderable(s) dropped", cameraCount, modelCount);
}

GpuResourceCensus AssetService::gpuResourceCensus() const
{
    return gpuResourceFactory.gpuResourceCensus();
}

} // namespace raceengine
