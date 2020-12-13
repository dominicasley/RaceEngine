
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

std::optional<Shader*> ShaderService::createShader(const std::string& name, const ShaderDescriptor& shaderDescriptor)
{
    auto shaderProgramId = openGlRenderer.createShaderObject(shaderDescriptor);

    if (!shaderProgramId.has_value())
    {
        return {};
    }

    return memoryStorageService.shaders.add(
        name,
        Shader{
            .gpuResourceId = shaderProgramId.value()
        });
}
