#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

struct SceneNode {
    SceneNode* parent = nullptr;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::mat4 rotation = glm::mat4(1.0f);
    glm::mat4 position = glm::mat4(1.0f);
    glm::mat4 scale = glm::mat4(1.0f);
    glm::vec3 forward = glm::vec3(0.0f);
};
