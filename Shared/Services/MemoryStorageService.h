#pragma once

#include <mutex>
#include <thread>
#include <memory_resource>
#include <array>
#include <string>
#include <spdlog/logger.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/animation/runtime/animation.h>
#include <Graphics/Models/Scene/Model.h>
#include <Graphics/Models/Scene/Texture.h>
#include <Graphics/Models/Scene/CubeMap.h>
#include <Graphics/Models/Scene/Material.h>
#include <Graphics/Models/Scene/Shader.h>
#include <Graphics/Models/Scene/Fbo.h>
#include <Graphics/Models/Scene/PostProcess.h>
#include <Shared/Types/Resource.h>

template<typename T>
class MemoryStorage
{
    mutable unsigned long long index = 0;
    mutable std::mutex accessorMutex;
    mutable std::array<T, 1024> items;

public:
    explicit MemoryStorage() = default;

    [[nodiscard]] const T& get(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        return items[key.id];
    };

    [[nodiscard]] T& get(const Resource<T>& key)
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        return items[key.id];
    };

    [[nodiscard]] bool exists(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        return index > key.id;
    };

    [[nodiscard]] std::optional<Resource<T>> getKeyIfExists(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);

        if (items.find(key) != items.end()) {
            return key;
        }

        return std::nullopt;
    };

    void remove(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        // items.erase(key.id);
        // todo: idek
    };

    Resource<T> add(const T& item) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        items[index] = item;
        index++;

        return Resource<T> {
            .id = index - 1,
            .value = &items[index - 1]
        };
    };

    void update(const Resource<T>& resource, T& value) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        items[resource.id] = value;
    };


    Resource<T> takeOwnership(T& item) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);

        items[index] = std::move(item);
        index++;

        return Resource<T> {
            .id = index - 1,
            .value = &items[index - 1]
        };
    };
};

class MemoryStorageService
{
private:
    spdlog::logger& logger;

public:
    const MemoryStorage<Model> models;
    const MemoryStorage<Mesh> meshes;
    const MemoryStorage<Texture> textures;
    const MemoryStorage<Material> materials;
    const MemoryStorage<CubeMap> cubeMaps;
    const MemoryStorage<Shader> shaders;
    const MemoryStorage<Fbo> frameBuffers;
    const MemoryStorage<FboAttachment> bufferAttachments;
    const MemoryStorage<PostProcess> postProcesses;
    const MemoryStorage<std::unique_ptr<ozz::animation::Skeleton>> skeletons;
    const MemoryStorage<std::unique_ptr<ozz::animation::Animation>> animations;

    explicit MemoryStorageService(spdlog::logger& logger);
};


