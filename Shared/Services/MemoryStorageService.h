#pragma once

#include <mutex>
#include <thread>
#include <deque>
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

// std::deque backing: growth never relocates elements, so cached Resource<T>::value pointers stay valid; removal would require generational handles.
template<typename T>
class MemoryStorage
{
    mutable std::mutex accessorMutex;
    mutable std::deque<T> items;

public:
    [[nodiscard]] const T& get(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        return items[key.id];
    };

    [[nodiscard]] bool exists(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        return items.size() > key.id;
    };

    Resource<T> add(const T& item) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        T& stored = items.emplace_back(item);

        return Resource<T> {
            .id = items.size() - 1,
            .value = &stored
        };
    };

    Resource<T> add(T&& item) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        T& stored = items.emplace_back(std::move(item));

        return Resource<T> {
            .id = items.size() - 1,
            .value = &stored
        };
    };

    void update(const Resource<T>& resource, T& value) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        items[resource.id] = value;
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


