#pragma once

#include <glm/mat4x4.hpp>
#include <string>

struct Joint
{
public:
    int id;
    Joint *parent;
    std::string name;
    glm::mat4 inverseBindPoseMatrix;
    glm::mat4 cachedTransform;

    glm::mat4 computeInverseBindPoseMatrix() {
        if (this->parent)
        {
            this->cachedTransform = this->parent->computeInverseBindPoseMatrix() * this->inverseBindPoseMatrix;
            return this->cachedTransform;
        }

        this->cachedTransform = this->inverseBindPoseMatrix;
        return this->cachedTransform;
    }
};
