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
    auto e = scenes.emplace(name, std::make_unique<Scene>());
    return e.first->second.get();
}

SceneNode* SceneManagerService::createNode(Scene* scene)
{
    auto& e = scene->nodes.emplace_back(std::make_unique<SceneNode>());
    return e.get();
}

void SceneManagerService::setPosition(SceneNode* node, float x, float y, float z) const
{
    node->position = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
}

void SceneManagerService::setDirection(SceneNode* node, float angle, float x, float y, float z) const
{
    node->rotation = glm::rotate(glm::mat4(1.0), angle, glm::vec3(x, y, z));
}

void SceneManagerService::setScale(SceneNode* node, float x, float y, float z) const
{
    node->scale = glm::scale(glm::mat4(1.0f), glm::vec3(x, y, z));
}

void SceneManagerService::translate(SceneNode* node, float x, float y, float z) const
{
    node->position = glm::translate(node->position, glm::vec3(x, y, z));
}

void SceneManagerService::rotate(SceneNode* node, float angle, float x, float y, float z) const
{
    node->rotation = glm::rotate(node->rotation, angle, glm::vec3(x, y, z));
}

void SceneManagerService::scale(SceneNode* node, float x, float y, float z) const
{
    node->scale = glm::scale(node->scale, glm::vec3(x, y, z));
}

void SceneManagerService::setParent(SceneNode* node, SceneNode* parent) const
{
    node->parent = parent;
}

const glm::mat4& SceneManagerService::modelMatrix(SceneNode* node) const
{
    node->modelMatrix = node->position * node->rotation * node->scale;

    if (node->parent)
    {
        node->modelMatrix = modelMatrix(node->parent) * node->modelMatrix;
    }

    node->forward = normalize(glm::vec3(glm::inverse(node->modelMatrix)[2]));

    return node->modelMatrix;
}