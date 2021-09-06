#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <future>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include "Shared/Services/MemoryStorageService.h"
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
    [[nodiscard]] std::string loadTextFile(const std::string& filePath) const;
    [[nodiscard]] Resource<Model> loadModel(const std::string& filePath) const;
    [[nodiscard]] Resource<std::unique_ptr<ozz::animation::Skeleton>> loadSkeleton(const std::string& filePath) const;
    [[nodiscard]] Resource<std::unique_ptr<ozz::animation::Animation>> loadAnimation(const std::string& filePath) const;
    [[nodiscard]] Resource<Texture> loadTexture(const std::string& filePath) const;
    [[nodiscard]] const Observable<std::string>& loadTextFileAsync(std::string filePath) const;
    [[nodiscard]] const Observable<Resource<Model>>& loadModelAsync(std::string filePath) const;
    [[nodiscard]] const Observable<Resource<Texture>>& loadTextureAsync(std::string filePath) const;
    [[nodiscard]] const Observable<Resource<std::unique_ptr<ozz::animation::Skeleton>>>& loadSkeletonAsync(std::string filePath) const;
    [[nodiscard]] const Observable<Resource<std::unique_ptr<ozz::animation::Animation>>>& loadAnimationAsync(std::string filePath) const;
};
