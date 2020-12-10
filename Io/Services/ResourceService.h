#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <future>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

#include "MemoryStorageService.h"
#include "Async.h"

class ResourceService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    BackgroundWorkerService& backgroundWorkerService;

public:
    explicit ResourceService(
        spdlog::logger& logger,
        MemoryStorageService& memoryStorageService,
        BackgroundWorkerService& backgroundWorkerService);
    [[nodiscard]] std::string loadTextFile(const std::string& filePath);
    [[nodiscard]] tinygltf::Model* loadModel(const std::string& filePath);
    [[nodiscard]] ozz::animation::Skeleton* loadSkeleton(const std::string& filePath);
    [[nodiscard]] ozz::animation::Animation* loadAnimation(const std::string& filePath);
    const Observable<std::string>& loadTextFileAsync(std::string filePath);
    const Observable<tinygltf::Model*>& loadModelAsync(std::string filePath);
    const Observable<ozz::animation::Skeleton*>& loadSkeletonAsync(std::string filePath);
    const Observable<ozz::animation::Animation*>& loadAnimationAsync(std::string filePath);
};
