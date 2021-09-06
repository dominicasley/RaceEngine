#pragma once

#include <spdlog/logger.h>

#include "FboService.h"
#include "Graphics/Models/Scene/Camera.h"
#include "Graphics/System/IWindow.h"

class CameraService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    IWindow& window;

public:
    explicit CameraService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window);
    Camera createCamera();
    void setPosition(Camera* camera, float x, float y, float z) const;
    void setDirection(Camera* camera, float x, float y, float z) const;
    void translate(Camera* camera,float x, float y, float z) const;
    void rotate(Camera* camera, float x, float y, float z) const;
    void setRoll(Camera* camera, float x, float y, float z) const;
    void setAspectRatio(Camera* camera, float aspectRatio) const;
    void recreateOutputBuffer(Camera* camera, int width, int height) const;
    void setFieldOfView(Camera* camera, float fov) const;
    void lookAtPoint(Camera* camera, float x, float y, float z) const;
    void addPostProcess(Camera* camera, const Resource<PostProcess>& postProcess) const;
    const Fbo& getOutputBuffer(Camera* camera) const;
    const glm::mat4& updateModelViewProjectionMatrix(Camera* camera) const;
    const glm::mat4& updateModelViewMatrix(Camera* camera) const;
};


