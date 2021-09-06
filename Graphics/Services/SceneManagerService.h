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
    [[nodiscard]] SceneNode* createNode(Scene* scene);
    [[nodiscard]] const glm::mat4& modelMatrix(SceneNode* node) const;
    void setPosition(SceneNode* node, float x, float y, float z) const;
    void setDirection(SceneNode* node, float angle, float x, float y, float z) const;
    void setScale(SceneNode* node, float x, float y, float z) const;
    void translate(SceneNode* node, float x, float y, float z) const;
    void rotate(SceneNode* node, float angle, float x, float y, float z) const;
    void scale(SceneNode* node, float x, float y, float z) const;
    void setParent(SceneNode* node, SceneNode* parent) const;

};


