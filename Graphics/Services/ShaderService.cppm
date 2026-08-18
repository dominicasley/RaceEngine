module;

#include <expected>
#include <map>
#include <optional>
#include <string>

export module raceengine.graphics:ShaderService;

import :IGpuResourceFactory;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class ShaderService
{
private:
    MemoryStorageService& memoryStorageService;
    IGpuResourceFactory& gpuResourceFactory;

    std::map<std::string, Resource<Shader>> shaders;

public:
    explicit ShaderService(MemoryStorageService& memoryStorageService, IGpuResourceFactory& gpuResourceFactory);
    [[nodiscard]] std::expected<Resource<Shader>, std::string> createShader(const std::string& name,
                                                                            const ShaderDescriptor& shaderDescriptor);
    // A name nothing was registered under is a lookup miss, not a failed operation: the
    // caller asked whether a shader exists and the answer is no.
    [[nodiscard]] std::optional<Resource<Shader>> getShaderByName(const std::string& name);
};

ShaderService::ShaderService(MemoryStorageService& memoryStorageService, IGpuResourceFactory& gpuResourceFactory) :
    memoryStorageService(memoryStorageService),
    gpuResourceFactory(gpuResourceFactory)
{
}

std::expected<Resource<Shader>, std::string> ShaderService::createShader(const std::string& name,
                                                                         const ShaderDescriptor& shaderDescriptor)
{
    const auto shaderProgramId = gpuResourceFactory.createShaderObject(shaderDescriptor);

    if (!shaderProgramId)
    {
        return std::unexpected("shader '" + name + "' was not created: " + shaderProgramId.error());
    }

    const auto shader = memoryStorageService.shaders.add(Shader{.gpuResourceId = shaderProgramId.value()});

    shaders[name] = shader;

    return shader;
}

std::optional<Resource<Shader>> ShaderService::getShaderByName(const std::string& name)
{
    if (shaders.contains(name))
        return shaders.at(name);

    return {};
}

} // namespace raceengine
