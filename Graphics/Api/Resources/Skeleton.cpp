#include "Skeleton.h"
#include <glm/mat4x4.hpp>
#include <utility>

Joint::Joint(std::string name, const unsigned int id, Joint *parent) :
        id(id),
        parent(parent),
        name(std::move(name))
{
}

Joint::~Joint()
{
}

Joint *Joint::getParent() const
{
    return this->parent;
}

void Joint::setInverseBindPoseMatrix(const glm::mat4 &matrix)
{
    this->inverseBindPoseMatrix = matrix;
}

glm::mat4 Joint::getInverseBindPoseMatrix() const
{
    return this->inverseBindPoseMatrix;
}

Skeleton::Skeleton(std::string name) :
        name(std::move(name))
{
}

void Skeleton::addJoint(const std::string &name, unsigned int id, const int parent)
{
    joints.push_back(std::make_unique<Joint>(name, id, parent != -1 ? this->getJointByIndex(parent) : nullptr));
}

std::vector<std::shared_ptr<Joint>> *Skeleton::getJoints()
{
    return &this->joints;
}

bool Skeleton::getJointIndexByName(const std::string &name, unsigned int *outIndex)
{
    for (auto &joint : this->joints)
    {
        if (joint->getName() == name)
        {
            *outIndex = joint->index();
            return true;
        }
    }

    return false;
}

Joint *Skeleton::getJointByIndex(const unsigned int index)
{
    for (auto &joint : this->joints)
    {
        if (joint->index() == index)
        {
            return joint.get();
        }
    }

    return nullptr;
}

size_t Skeleton::getSize() const
{
    return joints.size();
}

