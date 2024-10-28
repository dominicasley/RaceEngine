#pragma once

#include <spdlog/spdlog.h>
#include <memory_resource>


#include "../Models/Scene/Scene.h"

class SceneManagerService
{
private:
    spdlog::logger& logger;

    std::pmr::monotonic_buffer_resource sceneBufferResource;
    std::vector<Scene> sceneBuffer[1024];
    mutable std::pmr::vector<Scene> scenes;

public:
    explicit SceneManagerService(spdlog::logger& logger);

    [[nodiscard]] std::pmr::vector<Scene>& getScenes();
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


