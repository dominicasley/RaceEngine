#pragma once

#include <spdlog/logger.h>

#include "Async/Async.h"
#include "Graphics/Graphics.h"
#include "Io/Io.h"
#include "Shared/Shared.h"

class Bootstrapper
{
public:
    spdlog::logger& logger;
    IWindow& window;
    OpenGLRenderer& renderer;
    ResourceService& resourceService;
    MemoryStorageService& memoryStorageService;
    BackgroundWorkerService& backgroundWorkerService;
    SceneManagerService& sceneManager;
    SceneService& sceneService;
    RenderableEntityService& renderableEntityService;
    CameraService& cameraService;
    ShaderService& shaderService;
    CubeMapService& cubeMapService;

    Bootstrapper(
        spdlog::logger& logger,
        MemoryStorageService& memoryStorageService,
        ResourceService& resourceService,
        IWindow& window,
        OpenGLRenderer& renderer,
        BackgroundWorkerService& backgroundWorkerService,
        SceneManagerService& sceneManager,
        SceneService& sceneService,
        RenderableEntityService& renderableEntityService,
        CameraService& cameraService,
        ShaderService& shaderService,
        CubeMapService& cubeMapService);

    void step(float delta);
    void draw(float delta);
};
