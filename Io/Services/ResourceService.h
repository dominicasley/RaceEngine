#pragma once

#include <Logger.h>
#include <SigSlot.h>
#include <map>
#include <string>
#include <future>

#include "MemoryStorageService.h"
#include "Async.h"

class ResourceService
{
private:
    Logger& logger;
    MemoryStorageService& memoryStorageService;
    BackgroundWorkerService& backgroundWorkerService;

public:
    explicit ResourceService(
        Logger& logger,
        MemoryStorageService& memoryStorageService,
        BackgroundWorkerService& backgroundWorkerService);
    [[nodiscard]] tinygltf::Model* loadModel(std::string filePath);
    [[nodiscard]] std::string loadTextFile(std::string filePath);
    const Observable<tinygltf::Model*>& loadModelAsync(std::string filePath);
    const Observable<std::string>& loadTextFileAsync(std::string filePath);
};
