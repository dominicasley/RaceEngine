#pragma once
#define BOOST_DI_CFG_CTOR_LIMIT_SIZE 32

#include <boost/di.hpp>
#include "Bootstrapper.h"

import Game;
import Physics;

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
    RenderableEntityService& renderableEntity;
    CameraService& camera;
    ShaderService& shader;
    CubeMapService& cubeMap;
    FboService& fbo;
    PostProcessService& postProcess;
    PresenterService& presenter;
    EntityService& entity;
    PhysicsService& physics;

    Engine();
    [[nodiscard]] bool running() const;
    void step();
};