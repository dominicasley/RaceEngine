#include "CameraService.h"

CameraService::CameraService(spdlog::logger& logger) : logger(logger)
{
}

Camera CameraService::createCamera()
{
    return Camera {
        .iso = 6400,
        .aspectRatio = 16.0f / 9.0f,
        .aperture = 1.4f,
        .fieldOfView = 55.f,
        .direction = glm::vec3(0, 0, 1),
        .roll = glm::vec3(0, 1, 0)
    };
}

void CameraService::setPosition(Camera& camera, float x, float y, float z) const
{
    camera.position.x = x;
    camera.position.y = y;
    camera.position.z = z;
}

void CameraService::translate(Camera& camera, float x, float y, float z) const
{
    camera.position.x += x;
    camera.position.y += y;
    camera.position.z += z;
}

void CameraService::setDirection(Camera& camera, float x, float y, float z) const
{
    if (x == camera.position.x && y == camera.position.y && z == camera.position.z)
        return;

    camera.direction.x = x;
    camera.direction.y = y;
    camera.direction.z = z;
}

void CameraService::rotate(Camera& camera, float x, float y, float z) const
{
    if (x == camera.position.x && y == camera.position.y && z == camera.position.z)
        return;

    camera.direction.x += x;
    camera.direction.y += y;
    camera.direction.z += z;
}

void CameraService::setRoll(Camera& camera, float x, float y, float z) const
{
    camera.roll.x = x;
    camera.roll.y = y;
    camera.roll.z = z;
}

void CameraService::setAspectRatio(Camera& camera, float aspectRatio) const
{
    camera.aspectRatio = aspectRatio;
}

void CameraService::lookAtPoint(Camera& camera, float x, float y, float z) const
{
    const auto point = glm::vec3(x, y, z);
    camera.direction = glm::normalize(point - camera.position);
}

void CameraService::setFieldOfView(Camera& camera, float fov) const
{
    camera.fieldOfView = fov;
}

const glm::mat4& CameraService::updateModelViewMatrix(Camera& camera) const
{
    camera.modelViewMatrix = glm::lookAt(camera.position, camera.position + camera.direction, camera.roll);
    return camera.modelViewMatrix;
}

const glm::mat4& CameraService::updateModelViewProjectionMatrix(Camera& camera) const
{
    camera.modelViewProjectionMatrix =
        glm::perspective(glm::radians(camera.fieldOfView), camera.aspectRatio, 1.0f, 1000.0f) *
        updateModelViewMatrix(camera);

    return camera.modelViewProjectionMatrix;
}
