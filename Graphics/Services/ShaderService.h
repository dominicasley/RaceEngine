#pragma once


#include <spdlog/logger.h>
#include "Shared/Services/MemoryStorageService.h"
#include "../Models/Scene/ShaderDescriptor.h"
#include "../Models/Scene/Shader.h"
#include "../Api/OpenGLRenderer.h"

class ShaderService
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


