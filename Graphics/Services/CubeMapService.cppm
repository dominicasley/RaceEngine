module;

#include <expected>
#include <string>

export module raceengine.graphics:CubeMapService;

import :IGpuResourceFactory;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class CubeMapService
{
private:
    IGpuResourceFactory& gpuResourceFactory;
    MemoryStorageService& memoryStorageService;

public:
    explicit CubeMapService(IGpuResourceFactory& gpuResourceFactory, MemoryStorageService& memoryStorageService);
    [[nodiscard]] std::expected<Resource<CubeMap>, std::string>
    create(const std::string& name, const Resource<Texture>& front, const Resource<Texture>& back,
           const Resource<Texture>& left, const Resource<Texture>& right, const Resource<Texture>& top,
           const Resource<Texture>& bottom) const;
};

CubeMapService::CubeMapService(IGpuResourceFactory& gpuResourceFactory, MemoryStorageService& memoryStorageService) :
    gpuResourceFactory(gpuResourceFactory),
    memoryStorageService(memoryStorageService)
{
}

std::expected<Resource<CubeMap>, std::string>
CubeMapService::create(const std::string& name, const Resource<Texture>& frontTextureKey,
                       const Resource<Texture>& backTextureKey, const Resource<Texture>& leftTextureKey,
                       const Resource<Texture>& rightTextureKey, const Resource<Texture>& topTextureKey,
                       const Resource<Texture>& bottomTextureKey) const
{
    const auto gpuResourceId = gpuResourceFactory.createCubeMap(
        memoryStorageService.textures.get(frontTextureKey), memoryStorageService.textures.get(backTextureKey),
        memoryStorageService.textures.get(leftTextureKey), memoryStorageService.textures.get(rightTextureKey),
        memoryStorageService.textures.get(topTextureKey), memoryStorageService.textures.get(bottomTextureKey));

    if (!gpuResourceId)
    {
        return std::unexpected("cube map '" + name + "' was not created: " + gpuResourceId.error());
    }

    return memoryStorageService.cubeMaps.add(CubeMap{.gpuResourceId = gpuResourceId.value(),
                                                     .front = frontTextureKey,
                                                     .back = backTextureKey,
                                                     .left = leftTextureKey,
                                                     .right = rightTextureKey,
                                                     .top = topTextureKey,
                                                     .bottom = bottomTextureKey});
}

} // namespace raceengine
