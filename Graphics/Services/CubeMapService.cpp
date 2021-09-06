
#include "CubeMapService.h"

CubeMapService::CubeMapService(
    spdlog::logger& logger,
    OpenGLRenderer& renderer,
    MemoryStorageService& memoryStorageService) :
    logger(logger),
    renderer(renderer),
    memoryStorageService(memoryStorageService)
{

}

Resource<CubeMap> CubeMapService::create(
    const std::string& name,
    const Resource<Texture>& frontTextureKey,
    const Resource<Texture>& backTextureKey,
    const Resource<Texture>& leftTextureKey,
    const Resource<Texture>& rightTextureKey,
    const Resource<Texture>& topTextureKey,
    const Resource<Texture>& bottomTextureKey) const
{
    auto gpuResourceId = renderer.createCubeMap(
        memoryStorageService.textures.get(frontTextureKey),
        memoryStorageService.textures.get(backTextureKey),
        memoryStorageService.textures.get(leftTextureKey),
        memoryStorageService.textures.get(rightTextureKey),
        memoryStorageService.textures.get(topTextureKey),
        memoryStorageService.textures.get(bottomTextureKey)
    );

    return memoryStorageService.cubeMaps.add(CubeMap {
        .gpuResourceId = gpuResourceId,
        .front = frontTextureKey,
        .back = backTextureKey,
        .left = leftTextureKey,
        .right = rightTextureKey,
        .top = topTextureKey,
        .bottom = bottomTextureKey
    });
}
