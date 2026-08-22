// CubeMapService bodies. Declarations are in Graphics/Services/CubeMapService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <array>
#include <cstddef>
#include <expected>
#include <string>

module raceengine.graphics;

import :CubeMapService;
import :IGpuResourceFactory;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

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
    const std::array faceKeys = {frontTextureKey, backTextureKey, leftTextureKey,
                                 rightTextureKey, topTextureKey,  bottomTextureKey};

    std::array<const Texture*, 6> faces{};
    for (size_t face = 0; face < faceKeys.size(); face++)
    {
        faces[face] = memoryStorageService.textures.find(faceKeys[face]);
        if (faces[face] == nullptr)
        {
            return std::unexpected("cube map '" + name + "' names a texture that is no longer loaded");
        }
    }

    const auto gpuResourceId =
        gpuResourceFactory.createCubeMap(*faces[0], *faces[1], *faces[2], *faces[3], *faces[4], *faces[5]);

    if (!gpuResourceId)
    {
        return std::unexpected("cube map '" + name + "' was not created: " + gpuResourceId.error());
    }

    // The GPU owns the texels now, so the CPU copies go — here rather than inside the backend,
    // which used to const_cast the const Texture& it was handed to do it. The face textures are
    // the cube map's from this point on: it is what emptied them, and AssetService::unloadCubeMap
    // is what removes them.
    for (const auto& faceKey : faceKeys)
    {
        memoryStorageService.textures.mutate(faceKey,
                                             [](Texture& texture)
                                             {
                                                 texture.data.clear();
                                                 texture.data.shrink_to_fit();
                                             });
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
