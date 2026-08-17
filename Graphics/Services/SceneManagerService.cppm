module;

#include <deque>

#include <spdlog/logger.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Graphics/Models/Scene/Scene.h>

export module raceengine.graphics:SceneManagerService;

namespace raceengine
{

export class SceneManagerService
{
private:
    spdlog::logger& logger;

    std::deque<Scene> scenes;

public:
    explicit SceneManagerService(spdlog::logger& logger);

    [[nodiscard]] std::deque<Scene>& getScenes();
    [[nodiscard]] Scene& getScene(int index);
    [[nodiscard]] Scene& createScene();
    [[nodiscard]] SceneNode& createNode(Scene& scene);
    [[nodiscard]] const glm::mat4& modelMatrix(SceneNode& node) const;
    void setPosition(SceneNode& node, float x, float y, float z) const;
    void setDirection(SceneNode& node, float angle, float x, float y, float z) const;
    void setScale(SceneNode& node, float x, float y, float z) const;
    void translate(SceneNode& node, float x, float y, float z) const;
    void rotate(SceneNode& node, float angle, float x, float y, float z) const;
    void scale(SceneNode& node, float x, float y, float z) const;
    void setParent(SceneNode& node, SceneNode& parent) const;
};

SceneManagerService::SceneManagerService(spdlog::logger& logger) :
    logger(logger)
{
}

std::deque<Scene>& SceneManagerService::getScenes()
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
    node.transformDirty = true;
}

void SceneManagerService::setDirection(SceneNode& node, float angle, float x, float y, float z) const
{
    node.rotation = glm::vec4(x, y, z, angle);
    node.rotationMatrix = glm::rotate(glm::mat4(1.0), angle, glm::vec3(x, y, z));
    node.transformDirty = true;
}

void SceneManagerService::setScale(SceneNode& node, float x, float y, float z) const
{
    node.scale = glm::vec3(x, y, z);
    node.scaleMatrix = glm::scale(glm::mat4(1.0f), node.scale);
    node.transformDirty = true;
}

void SceneManagerService::translate(SceneNode& node, float x, float y, float z) const
{
    node.position += glm::vec3(x, y, z);
    node.translationMatrix = glm::translate(node.translationMatrix, glm::vec3(x, y, z));
    node.transformDirty = true;
}

void SceneManagerService::rotate(SceneNode& node, float angle, float x, float y, float z) const
{
    node.rotation += glm::vec4(x, y, z, angle);
    node.rotationMatrix = glm::rotate(node.rotationMatrix, glm::radians(angle), glm::vec3(x, y, z));
    node.transformDirty = true;
}

void SceneManagerService::scale(SceneNode& node, float x, float y, float z) const
{
    node.scale += glm::vec3(x, y, z);
    node.scaleMatrix = glm::scale(node.scaleMatrix, glm::vec3(x, y, z));
    node.transformDirty = true;
}

void SceneManagerService::setParent(SceneNode& node, SceneNode& parent) const
{
    node.parent = &parent;
    node.transformDirty = true;
}

const glm::mat4& SceneManagerService::modelMatrix(SceneNode& node) const
{
    const glm::mat4* parentMatrix = nullptr;
    auto parentVersion = 0u;

    if (node.parent)
    {
        parentMatrix = &modelMatrix(*node.parent);
        parentVersion = node.parent->transformVersion;
    }

    if (node.transformDirty || parentVersion != node.parentTransformVersion)
    {
        node.modelMatrix = node.translationMatrix * node.rotationMatrix * node.scaleMatrix;

        if (parentMatrix)
        {
            node.modelMatrix = *parentMatrix * node.modelMatrix;
        }

        node.forward = normalize(glm::vec3(glm::inverse(node.modelMatrix)[2]));
        node.transformDirty = false;
        node.parentTransformVersion = parentVersion;
        node.transformVersion++;
    }

    return node.modelMatrix;
}

} // namespace raceengine
