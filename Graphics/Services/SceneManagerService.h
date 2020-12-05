#pragma once

#include <spdlog/spdlog.h>
#include <map>
#include "../Models/Scene/Scene.h"

class SceneManagerService
{
private:
    spdlog::logger& logger;
    std::map<std::string, std::unique_ptr<Scene>> scenes;

public:
    explicit SceneManagerService(spdlog::logger& logger);

    [[nodiscard]] const std::map<std::string, std::unique_ptr<Scene>>& getScenes();
    [[nodiscard]] Scene* getScene(const std::string& name);
    [[nodiscard]] Scene* createScene(const std::string& name);
};


