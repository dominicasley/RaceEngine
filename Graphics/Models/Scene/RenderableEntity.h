#pragma once

#include <memory>
#include <map>
#include <optional>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include "./Skeleton.h"
#include "../../Api/Resources/RenderableEntityDesc.h"

struct RenderableEntity
{
    RenderableEntity* parent;
    tinygltf::Model* model;
    glm::mat4 modelMatrix;
    glm::mat4 rotation;
    glm::mat4 position;
    glm::mat4 scale;
    glm::vec3 forward;
    unsigned int currentAnimationIndex;
    std::map<unsigned int, std::map<float, glm::vec3>> position_state;
    std::map<unsigned int, std::map<float, glm::quat>> rotation_state;
    std::map<unsigned int, std::map<float, glm::vec3>> scaling_state;
};



