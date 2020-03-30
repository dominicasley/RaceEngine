#pragma once
#include <string>
#include <vector>
#include <map>
#include <glm/vec3.hpp>
#include <glm/ext/quaternion_float.hpp>

struct Animation
{
    std::string name;
    float totalAnimationTime;
    std::map<unsigned int, std::vector<std::pair<float, glm::vec3>>> position;
    std::map<unsigned int, std::vector<std::pair<float, glm::vec3>>> scale;
    std::map<unsigned int, std::vector<std::pair<float, glm::quat>>> rotation;
};
