#pragma once

#include <mutex>
#include <thread>
#include <memory_resource>
#include <array>
#include <string>
#include <spdlog/logger.h>
#include <tiny_gltf.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>

template<typename T>
class MemoryStorage
{
    std::vector<T> buffer[1024];
    std::pmr::monotonic_buffer_resource bufferResource;
    mutable std::mutex accessorMutex;
    mutable std::pmr::unordered_map<std::string, std::unique_ptr<T>> items;

public:
    MemoryStorage() :
        bufferResource(buffer, buffer->size()),
        items(std::pmr::unordered_map<std::string, std::unique_ptr<T>>(&bufferResource))
    {
        items.reserve(1024);
    }

    [[nodiscard]] T* get(const std::string& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        return items.at(key).get();
    };

    [[nodiscard]] bool exists(const std::string& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        return items.find(key) != items.end();
    };

    void remove(const std::string& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        items.erase(key);
    };

    T* add(const std::string& key, const T& item) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        items.insert_or_assign(key, std::make_unique<T>(item));

        return items[key].get();
    };

    T* takeOwnership(const std::string& key, std::unique_ptr<T>& ptr) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        items.insert_or_assign(key, std::move(ptr));

        return items[key].get();
    };
};

class MemoryStorageService
{
private:
    spdlog::logger& logger;

public:
    const MemoryStorage<tinygltf::Model> models;
    const MemoryStorage<ozz::animation::Skeleton> skeletons;
    const MemoryStorage<ozz::animation::Animation> animations;

    explicit MemoryStorageService(spdlog::logger& logger);
};


