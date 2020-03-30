#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <future>

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
    [[nodiscard]] tinygltf::Model* loadModel(std::string filePath);
    [[nodiscard]] std::string loadTextFile(std::string filePath);
    const Observable<tinygltf::Model*>& loadModelAsync(std::string filePath);
    const Observable<std::string>& loadTextFileAsync(std::string filePath);
};
