module;

#include <string>

#include <spdlog/logger.h>

#include <Shared/Services/MemoryStorageService.h>
#include <Graphics/Models/Scene/CubeMap.h>

export module raceengine.graphics:CubeMapService;

import :OpenGLRenderer;

namespace raceengine
{

export class CubeMapService
{
private:
    spdlog::logger& logger;
    OpenGLRenderer& renderer;
    MemoryStorageService& memoryStorageService;

public:
    explicit CubeMapService(spdlog::logger& logger, OpenGLRenderer& renderer, MemoryStorageService& memoryStorageService);
    Resource<CubeMap> create(
        const std::string& name,
        const Resource<Texture>& front,
        const Resource<Texture>& back,
        const Resource<Texture>& left,
        const Resource<Texture>& right,
        const Resource<Texture>& top,
        const Resource<Texture>& bottom) const;
};

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

} // namespace raceengine
