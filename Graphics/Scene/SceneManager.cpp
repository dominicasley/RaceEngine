//
// Created by Dominic on 2/01/2020.
//

#include "SceneManager.h"

SceneManager::SceneManager(Logger& logger) : logger(logger)
{

}

const std::map<std::string, std::unique_ptr<Scene>>& SceneManager::getScenes()
{
    return scenes;
}

Scene* SceneManager::getScene(const std::string& name)
{
    return scenes.at(name).get();
}

Scene* SceneManager::createScene(const std::string& name)
{
    scenes.emplace(name, std::make_unique<Scene>());
    return scenes[name].get();
}
