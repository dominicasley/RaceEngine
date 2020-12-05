#include "SceneManagerService.h"

SceneManagerService::SceneManagerService(spdlog::logger& logger) : logger(logger)
{

}

const std::map<std::string, std::unique_ptr<Scene>>& SceneManagerService::getScenes()
{
    return scenes;
}

Scene* SceneManagerService::getScene(const std::string& name)
{
    return scenes.at(name).get();
}

Scene* SceneManagerService::createScene(const std::string& name)
{
    scenes.emplace(name, std::make_unique<Scene>());
    return scenes[name].get();
}
