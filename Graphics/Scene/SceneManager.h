//
// Created by Dominic on 2/01/2020.
//

#pragma once

#include <spdlog/spdlog.h>
#include <map>

#include "Scene.h"

class SceneManager
{
private:
    spdlog::logger& logger;
    std::map<std::string, std::unique_ptr<Scene>> scenes;

public:
    explicit SceneManager(spdlog::logger& logger);

    [[nodiscard]] const std::map<std::string, std::unique_ptr<Scene>>& getScenes();
    [[nodiscard]] Scene* getScene(const std::string& name);
    [[nodiscard]] Scene* createScene(const std::string& name);
};


