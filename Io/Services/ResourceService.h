#pragma once

#include <spdlog/logger.h>
#include <expected>
#include <map>
#include <string>
#include <future>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include <Shared/Services/MemoryStorageService.h>
#include "GLTFService.h"
#include "Async/Async.h"

class ResourceService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    BackgroundWorkerService& backgroundWorkerService;
    GLTFService& gltfService;

public:
    explicit ResourceService(
        spdlog::logger& logger,
        MemoryStorageService& memoryStorageService,
        BackgroundWorkerService& backgroundWorkerService,
        GLTFService& gltfService);
    [[nodiscard]] std::expected<std::string, std::string> loadTextFile(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<Model>, std::string> loadModel(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<std::unique_ptr<ozz::animation::Skeleton>>, std::string> loadSkeleton(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<std::unique_ptr<ozz::animation::Animation>>, std::string> loadAnimation(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<Texture>, std::string> loadTexture(const std::string& filePath) const;
    [[nodiscard]] AsyncResult<std::string> loadTextFileAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<Model>> loadModelAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<Texture>> loadTextureAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<std::unique_ptr<ozz::animation::Skeleton>>> loadSkeletonAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<std::unique_ptr<ozz::animation::Animation>>> loadAnimationAsync(std::string filePath) const;
};
