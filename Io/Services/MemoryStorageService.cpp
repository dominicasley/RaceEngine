//
// Created by Dominic on 1/01/2020.
//

#include "MemoryStorageService.h"
#include <mutex>
#include <thread>

MemoryStorageService::MemoryStorageService(Logger& logger)
{

}

tinygltf::Model* MemoryStorageService::getModel(const std::string& key)
{
    std::lock_guard<std::mutex> lock(modelMutex);
    return models.at(key).get();
}

bool MemoryStorageService::modelExists(const std::string& key)
{
    std::lock_guard<std::mutex> lock(modelMutex);
    return models.find(key) != models.end();
}

void MemoryStorageService::removeModel(const std::string& key)
{
    std::lock_guard<std::mutex> lock(modelMutex);
    models.erase(key);
}

tinygltf::Model* MemoryStorageService::addModel(const std::string& key, const tinygltf::Model& model)
{
    std::lock_guard<std::mutex> lock(modelMutex);
    this->models[key] = std::make_unique<tinygltf::Model>(model);
    return this->models[key].get();
}
