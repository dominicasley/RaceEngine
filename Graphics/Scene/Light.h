#pragma once

#include <glm/vec3.hpp>


class Light
{
private:
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 color;
    float strength;

public:
    Light();
};

