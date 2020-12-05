#pragma once

#include <spdlog/logger.h>
#include "../Models/Scene/Camera.h"

class CameraService
{
private:
    spdlog::logger& logger;

public:
    explicit CameraService(spdlog::logger& logger);
    Camera createCamera();
    void setPosition(Camera& camera, float x, float y, float z) const;
    void setDirection(Camera& camera, float x, float y, float z) const;
    void translate(Camera& camera,float x, float y, float z) const;
    void rotate(Camera& camera, float x, float y, float z) const;
    void setRoll(Camera& camera, float x, float y, float z) const;
    void setAspectRatio(Camera& camera, float aspectRatio) const;
    void setFieldOfView(Camera& camera, float fov) const;
    void lookAtPoint(Camera& camera, float x, float y, float z) const;
    const glm::mat4& updateModelViewProjectionMatrix(Camera& camera) const;
    const glm::mat4& updateModelViewMatrix(Camera& camera) const;
};


