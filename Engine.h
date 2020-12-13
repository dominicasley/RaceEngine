#pragma once
#define BOOST_DI_CFG_CTOR_LIMIT_SIZE 16

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
    ShaderService& shader;

    Engine();
    [[nodiscard]] bool running() const;
    void step();
};