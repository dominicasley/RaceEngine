#pragma once

#include <memory>
#include <map>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include "../Api/Resources/RenderableEntityDesc.h"

class RenderableEntity
{
private:
    bool dirty;
    RenderableEntity* parent;
    tinygltf::Model* model;
    glm::mat4 modelMatrix;
    glm::mat4 rotation;
    glm::mat4 position;
    glm::mat4 scale;
    std::map<unsigned int, std::pair<float, unsigned int>> position_state;
    std::map<unsigned int, std::pair<float, unsigned int>> rotation_state;
    std::map<unsigned int, std::pair<float, unsigned int>> scaling_state;

public:
    RenderableEntity() : model(nullptr), parent(nullptr) {};
    explicit RenderableEntity(const RenderableEntityDesc&);

    [[nodiscard]] tinygltf::Model* getModel() const;
    void setPosition(float x, float y, float z);
    void setDirection(float angle, float x, float y, float z);
    void setScale(float x, float y, float z);

    void transform(float x, float y, float z);
    void rotate(float angle, float x, float y, float z);
    void resize(float x, float y, float z);

    void setParent(RenderableEntity* parent);
    void update(float delta);
    [[nodiscard]] const glm::mat4& getModelMatrix();
};



