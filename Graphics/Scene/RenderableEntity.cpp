#include "RenderableEntity.h"
#include "../Api/Resources/Mesh.h"
#include "../Api/Resources/RenderableEntityDesc.h"
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>


RenderableEntity::RenderableEntity(const RenderableEntityDesc &entityDesc) :
    model(entityDesc.model),
    dirty(true),
    parent(nullptr),
    position(glm::mat4(1.0f)),
    scale(glm::mat4(1.0f)),
    rotation(glm::mat4(1.0f))
{
}

tinygltf::Model *RenderableEntity::getModel() const
{
    return model;
}

void RenderableEntity::setPosition(float x, float y, float z)
{
    position = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
    dirty = true;
}

void RenderableEntity::setDirection(float angle, float x, float y, float z)
{
    rotation = glm::rotate(glm::mat4(1.0), angle, glm::vec3(x, y, z));
    dirty = true;
}

void RenderableEntity::setScale(const float x, const float y, const float z)
{
    scale = glm::scale(glm::mat4(1.0f), glm::vec3(x, y, z));
    dirty = true;
}

void RenderableEntity::transform(float x, float y, float z)
{
    position = glm::translate(position, glm::vec3(x, y, z));
    dirty = true;
}

void RenderableEntity::rotate(float angle, float x, float y, float z)
{
    rotation = glm::rotate(rotation, angle, glm::vec3(x, y, z));
    dirty = true;
}

void RenderableEntity::resize(float x, float y, float z)
{
    scale = glm::scale(scale, glm::vec3(x, y, z));
    dirty = true;
}

void RenderableEntity::update(const float delta)
{
}

const glm::mat4 &RenderableEntity::getModelMatrix()
{
    if (dirty)
    {
        modelMatrix = position *
                      rotation *
                      scale;

        if (parent)
        {
            modelMatrix = parent->getModelMatrix() * modelMatrix;
        }
    }

    return modelMatrix;
}

void RenderableEntity::setParent(RenderableEntity *_parent)
{
    this->parent = _parent;
}
