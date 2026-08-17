module;

#include <map>
#include <optional>
#include <string>

#include <spdlog/logger.h>

#include <Shared/Services/MemoryStorageService.h>
#include <Graphics/Models/Scene/ShaderDescriptor.h>
#include <Graphics/Models/Scene/Shader.h>

export module raceengine.graphics:ShaderService;

import :OpenGLRenderer;

namespace raceengine
{

export class ShaderService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    OpenGLRenderer& openGlRenderer;

    std::map<std::string, Resource<Shader>> shaders;

public:
    explicit ShaderService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, OpenGLRenderer& openGlRenderer);
    std::optional<Resource<Shader>> createShader(const std::string& name, const ShaderDescriptor& shaderDescriptor);
    std::optional<Resource<Shader>> getShaderByName(const std::string& name);
};

ShaderService::ShaderService(
    spdlog::logger& logger,
    MemoryStorageService& memoryStorageService,
    OpenGLRenderer& openGlRenderer) :
    logger(logger),
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
