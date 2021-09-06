#include "CameraService.h"

CameraService::CameraService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    window(window)
{
}

Camera CameraService::createCamera()
{
    auto windowState = window.state();

    auto windowWidth = static_cast<unsigned int>(windowState.windowWidth);
    auto windowHeight = static_cast<unsigned int>(windowState.windowHeight);

    return Camera {
        .iso = 6400,
        .aspectRatio = 16.0f / 9.0f,
        .aperture = 1.4f,
        .fieldOfView = 75.f,
        .direction = glm::vec3(0, 0, 1),
        .roll = glm::vec3(0, 1, 0),
        .output = fboService.create(CreateFboDTO {
            .type = FboType::Planar,
            .attachments = {
                CreateFboAttachmentDTO {
                    .width = windowWidth,
                    .height = windowHeight,
                    .type = FboAttachmentType::Color,
                    .captureFormat = TextureFormat::RGBA,
                    .internalFormat = TextureFormat::RGBA16F
                },
                CreateFboAttachmentDTO {
                    .width = windowWidth,
                    .height = windowHeight,
                    .type = FboAttachmentType::Depth,
                    .captureFormat = TextureFormat::DepthComponent,
                    .internalFormat = TextureFormat::DepthComponent
                },
            }
        })
    };
}

void CameraService::setPosition(Camera* camera, float x, float y, float z) const
{
    camera->position.x = x;
    camera->position.y = y;
    camera->position.z = z;
}

void CameraService::translate(Camera* camera, float x, float y, float z) const
{
    camera->position.x += x;
    camera->position.y += y;
    camera->position.z += z;
}

void CameraService::setDirection(Camera* camera, float x, float y, float z) const
{
    if (x == camera->position.x && y == camera->position.y && z == camera->position.z)
        return;

    camera->direction.x = x;
    camera->direction.y = y;
    camera->direction.z = z;
}

void CameraService::rotate(Camera* camera, float x, float y, float z) const
{
    if (x == camera->position.x && y == camera->position.y && z == camera->position.z)
        return;

    camera->direction.x += x;
    camera->direction.y += y;
    camera->direction.z += z;
}

void CameraService::setRoll(Camera* camera, float x, float y, float z) const
{
    camera->roll.x = x;
    camera->roll.y = y;
    camera->roll.z = z;
}

void CameraService::setAspectRatio(Camera* camera, float aspectRatio) const
{
    camera->aspectRatio = aspectRatio;
}

void CameraService::recreateOutputBuffer(Camera* camera, int width, int height) const
{
    auto windowWidth = static_cast<unsigned int>(width);
    auto windowHeight = static_cast<unsigned int>(height);

    fboService.resize(camera->output.value(), windowWidth, windowHeight);
}

void CameraService::lookAtPoint(Camera* camera, float x, float y, float z) const
{
    const auto point = glm::vec3(x, y, z);
    camera->direction = glm::normalize(point - camera->position);
}

void CameraService::setFieldOfView(Camera* camera, float fov) const
{
    camera->fieldOfView = fov;
}

const glm::mat4& CameraService::updateModelViewMatrix(Camera* camera) const
{
    camera->modelViewMatrix = glm::lookAt(camera->position, camera->position + camera->direction, camera->roll);
    return camera->modelViewMatrix;
}

const glm::mat4& CameraService::updateModelViewProjectionMatrix(Camera* camera) const
{
    camera->modelViewProjectionMatrix =
        glm::perspective(glm::radians(camera->fieldOfView), camera->aspectRatio, 1.0f, 5000.0f) *
        updateModelViewMatrix(camera);

    return camera->modelViewProjectionMatrix;
}

void CameraService::addPostProcess(Camera* camera, const Resource<PostProcess>& postProcessKey) const
{
    camera->postProcesses.push_back(postProcessKey);
}

const Fbo& CameraService::getOutputBuffer(Camera* camera) const {
    return memoryStorageService.frameBuffers.get(camera->output.value());
}
