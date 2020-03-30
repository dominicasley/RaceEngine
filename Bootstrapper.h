#pragma once

#include "Logger.h"
#include "Async/Async.h"
#include "Graphics/Graphics.h"
#include "Io/Io.h"

class Bootstrapper
{
public:
    Logger& logger;
    IWindow& window;
    OpenGLRenderer& renderer;
    ResourceService& resourceService;
    MemoryStorageService& memoryStorageService;
    BackgroundWorkerService& backgroundWorkerService;
    SceneManager& sceneManager;

    Bootstrapper(
            Logger& logger,
            MemoryStorageService& memoryStorageService,
            ResourceService& resourceService,
            IWindow& window,
            OpenGLRenderer& renderer,
            BackgroundWorkerService& backgroundWorkerService,
            SceneManager& sceneManager);

    void step();
    void draw();
};
