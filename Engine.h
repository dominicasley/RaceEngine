#pragma once

#include <memory>
#include <spdlog/logger.h>

#include "Async/Async.h"
#include "Game/Game.h"
#include "Graphics/Graphics.h"
#include "Io/Io.h"
#include "Shared/Shared.h"

class Engine
{
private:
    // Declaration order IS initialization order and reverse-destruction order:
    // each member may only depend on members declared above it.
    std::shared_ptr<spdlog::logger> logger;
    GLFWWindow glfwWindow;
    MemoryStorageService memoryStorageService;
    BackgroundWorkerService backgroundWorkerService;
    GLTFService gltfService;
    ResourceService resourceService;
    RenderableEntityService renderableEntityService;
    SceneManagerService sceneManagerService;
    OpenGLRenderer openGlRenderer;
    FboService fboService;
    ShaderService shaderService;
    CubeMapService cubeMapService;
    PostProcessService postProcessService;
    PresenterService presenterService;
    CameraService cameraService;
    SceneService sceneService;
    EntityService entityService;

public:
    Engine();
    [[nodiscard]] bool running() const;
    void step();

    [[nodiscard]] IWindow& window()
    {
        return glfwWindow;
    }

    [[nodiscard]] ResourceService& resource()
    {
        return resourceService;
    }

    [[nodiscard]] MemoryStorageService& memoryStorage()
    {
        return memoryStorageService;
    }

    [[nodiscard]] SceneManagerService& sceneManager()
    {
        return sceneManagerService;
    }

    [[nodiscard]] SceneService& scene()
    {
        return sceneService;
    }

    [[nodiscard]] RenderableEntityService& renderableEntity()
    {
        return renderableEntityService;
    }

    [[nodiscard]] CameraService& camera()
    {
        return cameraService;
    }

    [[nodiscard]] ShaderService& shader()
    {
        return shaderService;
    }

    [[nodiscard]] CubeMapService& cubeMap()
    {
        return cubeMapService;
    }

    [[nodiscard]] FboService& fbo()
    {
        return fboService;
    }

    [[nodiscard]] PostProcessService& postProcess()
    {
        return postProcessService;
    }

    [[nodiscard]] PresenterService& presenter()
    {
        return presenterService;
    }

    [[nodiscard]] EntityService& entity()
    {
        return entityService;
    }

private:
    void dumpFrameIfRequested();
};
