//
// Created by Dominic on 1/01/2020.
//

#pragma once

#include <unordered_map>
#include <string>
#include <spdlog/logger.h>
#include <tiny_gltf.h>

class MemoryStorageService
{
private:
    spdlog::logger& logger;
    std::mutex modelMutex;
    std::unordered_map<std::string, std::unique_ptr<tinygltf::Model>> models;

public:
    explicit MemoryStorageService(spdlog::logger& logger);
    [[nodiscard]] tinygltf::Model* getModel(const std::string& key);
    [[nodiscard]] bool modelExists(const std::string& key);
    void removeModel(const std::string& key);
    tinygltf::Model* addModel(const std::string& key, const tinygltf::Model& model);
};


