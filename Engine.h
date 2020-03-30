#pragma once

#include <boost/di.hpp>
#include "Bootstrapper.h"

namespace di = boost::di;

class Engine {
private:
    Bootstrapper app;

public:
    IWindow& window;
    OpenGLRenderer& renderer;
    ResourceService& resourceService;
    MemoryStorageService& memoryStorageService;
    BackgroundWorkerService& backgroundWorkerService;
    SceneManager& sceneManager;

    Engine();
    [[nodiscard]] bool running() const;
    void step();
};