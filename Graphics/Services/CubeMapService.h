#pragma once

#include <spdlog/logger.h>
#include <../../Shared/Services/MemoryStorageService.h>

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
    CubeMap* create(const std::string& name, Texture* front, Texture* back, Texture* left, Texture* right, Texture* top, Texture* bottom) const;
};


