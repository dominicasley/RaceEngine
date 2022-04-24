
#include "ShaderService.h"

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
