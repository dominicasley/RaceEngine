#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/mat4x4.hpp>

class Joint
{
private:
    unsigned int id;
    Joint *parent;
    std::string name;
    glm::mat4 inverseBindPoseMatrix{};

public:
    Joint(std::string, unsigned int, Joint *);

    ~Joint();

    std::string getName() const
    { return name; }

    unsigned int index() const
    { return id; }

    Joint *getParent() const;

    void setInverseBindPoseMatrix(const glm::mat4 &matrix);

    glm::mat4 getInverseBindPoseMatrix() const;
};

class Skeleton
{
private:
    std::string name;
    std::vector<std::shared_ptr<Joint>> joints;

public:
    Skeleton(std::string);

    void addJoint(const std::string &, unsigned int, int);

    std::vector<std::shared_ptr<Joint>> *getJoints();

    bool getJointIndexByName(const std::string &, unsigned int *outIndex);

    Joint *getJointByIndex(unsigned int index);

    [[nodiscard]] size_t getSize() const;
};
