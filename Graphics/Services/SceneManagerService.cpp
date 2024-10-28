#include "SceneManagerService.h"

SceneManagerService::SceneManagerService(spdlog::logger& logger) :
    logger(logger),
    sceneBufferResource(sceneBuffer, sceneBuffer->size()),
    scenes(std::pmr::vector<Scene>(&sceneBufferResource))
{
    scenes.reserve(1024);
}

std::pmr::vector<Scene>& SceneManagerService::getScenes()
{
    return scenes;
}

Scene& SceneManagerService::getScene(int index)
{
    return scenes.at(index);
}

Scene& SceneManagerService::createScene()
{
    auto& e =  scenes.emplace_back();
    return e;
}

SceneNode& SceneManagerService::createNode(Scene& scene)
{
    auto& e = scene.nodes.emplace_back();
    return e;
}

void SceneManagerService::setPosition(SceneNode& node, float x, float y, float z) const
{
    node.position = glm::vec3(x, y, z);
    node.translationMatrix = glm::translate(glm::mat4(1.0f), node.position);
}

void SceneManagerService::setDirection(SceneNode& node, float angle, float x, float y, float z) const
{
    node.rotation = glm::vec4(x, y, z, angle);
    node.rotationMatrix = glm::rotate(glm::mat4(1.0), angle, glm::vec3(x, y, z));
}

void SceneManagerService::setScale(SceneNode& node, float x, float y, float z) const
{
    node.scale = glm::vec3(x, y, z);
    node.scaleMatrix = glm::scale(glm::mat4(1.0f), node.scale);
}

void SceneManagerService::translate(SceneNode& node, float x, float y, float z) const
{
    node.position += glm::vec3(x, y, z);
    node.translationMatrix = glm::translate(node.translationMatrix, glm::vec3(x, y, z));
}

void SceneManagerService::rotate(SceneNode& node, float angle, float x, float y, float z) const
{
    node.rotation += glm::vec4(x, y, z, angle);
    node.rotationMatrix = glm::rotate(node.rotationMatrix, glm::radians(angle), glm::vec3(x, y, z));
}

void SceneManagerService::scale(SceneNode& node, float x, float y, float z) const
{
    node.scale += glm::vec3(x, y, z);
    node.scaleMatrix = glm::scale(node.scaleMatrix, glm::vec3(x, y, z));
}

void SceneManagerService::setParent(SceneNode& node, SceneNode& parent) const
{
    node.parent = &parent;
}

const glm::mat4& SceneManagerService::modelMatrix(SceneNode& node) const
{
    node.modelMatrix = node.translationMatrix * node.rotationMatrix * node.scaleMatrix;

    if (node.parent)
    {
        node.modelMatrix = modelMatrix(*node.parent) * node.modelMatrix;
    }

    node.forward = normalize(glm::vec3(glm::inverse(node.modelMatrix)[2]));

    return node.modelMatrix;
}