#pragma once

#include <spdlog/logger.h>

#include "Shared/Services/MemoryStorageService.h"
#include "../Models/Scene/CubeMap.h"
#include "../Api/OpenGLRenderer.h"

class CubeMapService
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


