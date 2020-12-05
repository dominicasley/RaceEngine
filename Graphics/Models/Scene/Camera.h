#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


struct Camera
{
    unsigned int iso;
    float aspectRatio;
    float aperture;
    float exposure;
    float fieldOfView;
    float nearClippingPlane;
    float farClippingPlane;
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 roll;
    glm::mat4 modelViewProjectionMatrix;
    glm::mat4 modelViewMatrix;
};

