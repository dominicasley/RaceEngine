#pragma once

#include <spdlog/logger.h>
#include <ranges>

#include "Shared/Services/MemoryStorageService.h"
#include "../Models/Scene/Fbo.h"
#include "../Models/Scene/CreateFboDTO.h"
#include "../Api/OpenGLRenderer.h"

class FboService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    OpenGLRenderer& renderer;

public:
    explicit FboService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, OpenGLRenderer& renderer);
    [[nodiscard]] Resource<Fbo> create(const CreateFboDTO& createFboDTO) const;
    void recreate(const Resource<Fbo>& fbo) const;
    void resize(const Resource<Fbo>& fbo, unsigned int width, unsigned int height) const;
    void deleteFbo(const Resource<Fbo>& fbo) const;

    [[nodiscard]] std::vector<Resource<FboAttachment>> getAttachmentsOfType(const Fbo& fbo, FboAttachmentType type) const;
};


