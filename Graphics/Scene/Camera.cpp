#include "Camera.h"

Camera::Camera() :
    fieldOfView(55.f),
    direction(0, 0, 1),
    roll(0, 1, 0),
    aspectRatio(16.0f / 9.0f),
    dirty(true),
    iso(6400),
    aperture(1.4)
{
}

void Camera::setPosition(float x, float y, float z)
{
    position.x = x;
    position.y = y;
    position.z = z;

    dirty = true;
}

void Camera::setDirection(float x, float y, float z)
{
    if (x == position.x && y == position.y && z == position.z)
    {
        return;
    }

    direction.x = x;
    direction.y = y;
    direction.z = z;

    dirty = true;
}

void Camera::setRoll(float x, float y, float z)
{
    roll.x = x;
    roll.y = y;
    roll.z = z;

    dirty = true;
}

void Camera::setAspectRatio(float _aspectRatio)
{
    this->aspectRatio = _aspectRatio;
    dirty = true;
}

void Camera::lookAtPoint(float x, float y, float z)
{
    const auto point = glm::vec3(x, y, z);
    direction = glm::normalize(point - position);

    dirty = true;
}

void Camera::setFieldOfView(const float fov)
{
    fieldOfView = fov;
    dirty = true;
}

void Camera::update()
{
    if (dirty)
    {
        mv = glm::lookAt(position, direction, roll);
        mvp = glm::perspective(glm::radians(fieldOfView), aspectRatio, 1.0f, 1000.0f) * mv;

        dirty = false;
    }
}

const glm::vec3 &Camera::getPosition() const
{
    return this->position;
}

const glm::mat4 &Camera::getMvpMatrix()
{
    update();
    return this->mvp;
}

const glm::mat4 &Camera::getMvMatrix()
{
    update();
    return this->mv;
}
