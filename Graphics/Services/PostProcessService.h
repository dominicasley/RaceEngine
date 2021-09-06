#pragma once

#include <spdlog/logger.h>

#include "FboService.h"
#include "Graphics/System/IWindow.h"
#include "Shared/Services/MemoryStorageService.h"

class PostProcessService
{
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    IWindow& window;

public:
    explicit PostProcessService(
        spdlog::logger& logger, MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window);
    [[nodiscard]] Resource<PostProcess> create(const std::string& id, const Resource<Shader>& shader) const;
    void addInput(const Resource<PostProcess>& postProcess, const Resource<FboAttachment>& attachment) const;
    void recreateOutputBuffer(const Resource<PostProcess> & postProcess, int width, int height) const;
};


