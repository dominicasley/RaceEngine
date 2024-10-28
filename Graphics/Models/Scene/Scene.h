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
    std::pmr::monotonic_buffer_resource cameraBufferResource;
    std::pmr::monotonic_buffer_resource lightBufferResource;
    std::pmr::monotonic_buffer_resource modelBufferResource;
    std::pmr::monotonic_buffer_resource nodeBufferResource;

    std::vector<Camera> cameraBuffer[1024];
    std::vector<Light> lightBuffer[1024];
    std::vector<RenderableModel> modelBuffer[1024];
    std::vector<SceneNode> nodeBuffer[1024];

    mutable std::pmr::vector<Camera> cameras;
    mutable std::pmr::vector<Light> lights;
    mutable std::pmr::vector<RenderableModel> models;
    mutable std::pmr::vector<SceneNode> nodes;

    explicit Scene() :
        cameraBufferResource(cameraBuffer, cameraBuffer->size()),
        cameras(std::pmr::vector<Camera>(&cameraBufferResource)),
        // lights
        lightBufferResource(lightBuffer, lightBuffer->size()),
        lights(std::pmr::vector<Light>(&lightBufferResource)),
        // models
        modelBufferResource(modelBuffer, modelBuffer->size()),
        models(std::pmr::vector<RenderableModel>(&modelBufferResource)),
        // nodes
        nodeBufferResource(nodeBuffer, nodeBuffer->size()),
        nodes(std::pmr::vector<SceneNode>(&nodeBufferResource))
    {
        cameras.reserve(1024);
        lights.reserve(1024);
        models.reserve(1024);
        nodes.reserve(1024);
    }

    Scene(Scene&& scene) noexcept:
        cameraBufferResource(cameraBuffer, cameraBuffer->size()),
        cameras(std::pmr::vector<Camera>(&cameraBufferResource)),
        // lights
        lightBufferResource(lightBuffer, lightBuffer->size()),
        lights(std::pmr::vector<Light>(&lightBufferResource)),
        // models
        modelBufferResource(modelBuffer, modelBuffer->size()),
        models(std::pmr::vector<RenderableModel>(&modelBufferResource)),
        // nodes
        nodeBufferResource(nodeBuffer, nodeBuffer->size()),
        nodes(std::pmr::vector<SceneNode>(&nodeBufferResource))
    {
        cameras.reserve(1024);
        lights.reserve(1024);
        models.reserve(1024);
        nodes.reserve(1024);
    }
};

