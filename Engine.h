#pragma once

#include <boost/di.hpp>
#include "Bootstrapper.h"

namespace di = boost::di;

class Engine
{
private:
    Bootstrapper app;

public:
    IWindow& window;
    OpenGLRenderer& renderer;
    ResourceService& resource;
    MemoryStorageService& memoryStorage;
    BackgroundWorkerService& backgroundWorker;
    SceneManagerService& sceneManager;
    SceneService& scene;
    RenderableEntityService& entity;
    CameraService& camera;

    Engine();
    [[nodiscard]] bool running() const;
    void step();
};