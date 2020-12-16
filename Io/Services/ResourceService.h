#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <future>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include "Services/MemoryStorageService.h"
#include "GLTFService.h"
#include "Async.h"

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
    [[nodiscard]] std::string loadTextFile(const std::string& filePath) const;
    [[nodiscard]] Model* loadModel(const std::string& filePath) const;
    [[nodiscard]] ozz::animation::Skeleton* loadSkeleton(const std::string& filePath) const;
    [[nodiscard]] ozz::animation::Animation* loadAnimation(const std::string& filePath) const;
    [[nodiscard]] Texture* loadTexture(const std::string& filePath) const;
    [[nodiscard]] const Observable<std::string>& loadTextFileAsync(std::string filePath) const;
    [[nodiscard]] const Observable<Model*>& loadModelAsync(std::string filePath) const;
    [[nodiscard]] const Observable<Texture*>& loadTextureAsync(std::string filePath) const;
    [[nodiscard]] const Observable<ozz::animation::Skeleton*>& loadSkeletonAsync(std::string filePath) const;
    [[nodiscard]] const Observable<ozz::animation::Animation*>& loadAnimationAsync(std::string filePath) const;
};
