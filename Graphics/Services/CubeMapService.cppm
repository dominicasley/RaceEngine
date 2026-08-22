module;

#include <array>
#include <cstddef>
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

} // namespace raceengine
