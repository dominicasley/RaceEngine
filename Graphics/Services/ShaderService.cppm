module;

#include <map>
#include <optional>
#include <string>

export module raceengine.graphics:ShaderService;

import :OpenGLRenderer;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class ShaderService
{
private:
    MemoryStorageService& memoryStorageService;
    OpenGLRenderer& openGlRenderer;

    std::map<std::string, Resource<Shader>> shaders;

public:
    explicit ShaderService(MemoryStorageService& memoryStorageService, OpenGLRenderer& openGlRenderer);
    std::optional<Resource<Shader>> createShader(const std::string& name, const ShaderDescriptor& shaderDescriptor);
    std::optional<Resource<Shader>> getShaderByName(const std::string& name);
};

ShaderService::ShaderService(
    MemoryStorageService& memoryStorageService,
    OpenGLRenderer& openGlRenderer) :
    memoryStorageService(memoryStorageService),
    openGlRenderer(openGlRenderer)
{

}

std::optional<Resource<Shader>>
ShaderService::createShader(const std::string& name, const ShaderDescriptor& shaderDescriptor)
{
    auto shaderProgramId = openGlRenderer.createShaderObject(shaderDescriptor);

    if (!shaderProgramId.has_value())
    {
        return {};
    }

    const auto shader = memoryStorageService.shaders.add(
        Shader {
            .gpuResourceId = shaderProgramId.value()
        });

    shaders[name] = shader;

    return shader;
}

std::optional<Resource<Shader>>
ShaderService::getShaderByName(const std::string& name)
{
    if (shaders.contains(name))
        return shaders.at(name);

    return {};
}

} // namespace raceengine
