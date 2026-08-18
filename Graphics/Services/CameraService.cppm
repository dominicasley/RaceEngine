module;

#include <expected>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module raceengine.graphics:CameraService;

import :FboService;
import :Window;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class CameraService
{
private:
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    IWindow& window;

public:
    explicit CameraService(MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window);
    [[nodiscard]] std::expected<Camera, std::string> createCamera();
    void setPosition(Camera& camera, float x, float y, float z) const;
    void setDirection(Camera& camera, float x, float y, float z) const;
    void translate(Camera& camera, float x, float y, float z) const;
    void rotate(Camera& camera, float x, float y, float z) const;
    void setRoll(Camera& camera, float x, float y, float z) const;
    void setAspectRatio(Camera& camera, float aspectRatio) const;
    void setClippingPlanes(Camera& camera, float nearPlane, float farPlane) const;
    [[nodiscard]] std::expected<void, std::string> recreateOutputBuffer(Camera& camera, int width, int height) const;
    void setFieldOfView(Camera& camera, float fov) const;
    void lookAtPoint(Camera& camera, float x, float y, float z) const;
    void addPostProcess(Camera& camera, const Resource<PostProcess>& postProcess) const;
    const Fbo& getOutputBuffer(Camera& camera) const;
    const glm::mat4& updateModelViewProjectionMatrix(Camera& camera) const;
    const glm::mat4& updateModelViewMatrix(Camera& camera) const;
};

CameraService::CameraService(MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window) :
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    window(window)
{
}

std::expected<Camera, std::string> CameraService::createCamera()
{
    auto windowState = window.state();

    auto windowWidth = static_cast<unsigned int>(windowState.windowWidth);
    auto windowHeight = static_cast<unsigned int>(windowState.windowHeight);

    // A camera is the render target it draws into; one that could not get its buffers is not a
    // camera with a missing field, so it is not handed back at all.
    const auto output =
        fboService.create(CreateFboDTO{.type = FboType::Planar,
                                       .attachments = {
                                           CreateFboAttachmentDTO{.width = windowWidth,
                                                                  .height = windowHeight,
                                                                  .type = FboAttachmentType::Color,
                                                                  .captureFormat = TextureFormat::RGBA,
                                                                  .internalFormat = TextureFormat::RGBA16F},
                                           CreateFboAttachmentDTO{.width = windowWidth,
                                                                  .height = windowHeight,
                                                                  .type = FboAttachmentType::Depth,
                                                                  .captureFormat = TextureFormat::DepthComponent,
                                                                  .internalFormat = TextureFormat::DepthComponent},
                                       }});

    if (!output)
    {
        return std::unexpected("the camera has no output buffer: " + output.error());
    }

    // The clipping planes default to the values the projection used to hardcode, so wiring the
    // fields changes no frame; a game that wants a different depth range now has one to set.
    return Camera{.iso = 6400,
                  .aspectRatio = 16.0f / 9.0f,
                  .aperture = 1.4f,
                  .fieldOfView = 75.f,
                  .nearClippingPlane = 1.0f,
                  .farClippingPlane = 5000.0f,
                  .direction = glm::vec3(0, 0, 1),
                  .roll = glm::vec3(0, 1, 0),
                  .output = output.value()};
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
    camera.direction.x = x;
    camera.direction.y = y;
    camera.direction.z = z;
}

// x, y and z are a delta applied to the current direction, not a direction: there is no
// predicate over them that says the camera would not move, so this writes unconditionally.
void CameraService::rotate(Camera& camera, float x, float y, float z) const
{
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

// Named nearPlane/farPlane, not near/far: those are still macros in the Windows headers the
// window layer pulls in.
void CameraService::setClippingPlanes(Camera& camera, float nearPlane, float farPlane) const
{
    camera.nearClippingPlane = nearPlane;
    camera.farClippingPlane = farPlane;
}

std::expected<void, std::string> CameraService::recreateOutputBuffer(Camera& camera, int width, int height) const
{
    auto windowWidth = static_cast<unsigned int>(width);
    auto windowHeight = static_cast<unsigned int>(height);

    return fboService.resize(camera.output.value(), windowWidth, windowHeight);
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
    camera.modelViewProjectionMatrix = glm::perspective(glm::radians(camera.fieldOfView), camera.aspectRatio,
                                                        camera.nearClippingPlane, camera.farClippingPlane) *
                                       updateModelViewMatrix(camera);

    return camera.modelViewProjectionMatrix;
}

void CameraService::addPostProcess(Camera& camera, const Resource<PostProcess>& postProcessKey) const
{
    camera.postProcesses.push_back(postProcessKey);
}

const Fbo& CameraService::getOutputBuffer(Camera& camera) const
{
    return memoryStorageService.frameBuffers.get(camera.output.value());
}

} // namespace raceengine
