#pragma once

#include <memory_resource>
#include <memory>

#include "Camera.h"
#include "Light.h"
#include "SceneNode.h"
#include "RenderableEntity.h"
#include "RenderableModel.h"

struct Scene
{
    // Heap-upstream monotonic arenas: the previous in-place buffers were built from an
    // uninitialized size read and overlapped live objects (heap corruption under clang).
    std::pmr::monotonic_buffer_resource cameraBufferResource;
    std::pmr::monotonic_buffer_resource lightBufferResource;
    std::pmr::monotonic_buffer_resource modelBufferResource;
    std::pmr::monotonic_buffer_resource nodeBufferResource;

    mutable std::pmr::vector<Camera> cameras;
    mutable std::pmr::vector<Light> lights;
    mutable std::pmr::vector<RenderableModel> models;
    mutable std::pmr::vector<SceneNode> nodes;

    explicit Scene() :
        cameras(std::pmr::vector<Camera>(&cameraBufferResource)),
        lights(std::pmr::vector<Light>(&lightBufferResource)),
        models(std::pmr::vector<RenderableModel>(&modelBufferResource)),
        nodes(std::pmr::vector<SceneNode>(&nodeBufferResource))
    {
        cameras.reserve(1024);
        lights.reserve(1024);
        models.reserve(1024);
        nodes.reserve(1024);
    }

    Scene(Scene&& scene) noexcept:
        cameras(std::pmr::vector<Camera>(&cameraBufferResource)),
        lights(std::pmr::vector<Light>(&lightBufferResource)),
        models(std::pmr::vector<RenderableModel>(&modelBufferResource)),
        nodes(std::pmr::vector<SceneNode>(&nodeBufferResource))
    {
        cameras.reserve(1024);
        lights.reserve(1024);
        models.reserve(1024);
        nodes.reserve(1024);
    }
};

