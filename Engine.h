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
    RenderableEntityService& renderableEntity;
    CameraService& camera;
    ShaderService& shader;
    CubeMapService& cubeMap;
    FboService& fbo;
    PostProcessService& postProcess;
    PresenterService& presenter;
    EntityService& entity;

    Engine();
    [[nodiscard]] bool running() const;
    void step();
};