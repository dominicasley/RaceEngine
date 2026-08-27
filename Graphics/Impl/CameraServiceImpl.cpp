// CameraService bodies. Declarations are in Graphics/Services/CameraService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

module raceengine.graphics;

import :CameraService;
import :FboService;
import :PhysicalCamera;
import :Window;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

CameraService::CameraService(MemoryStorageService& memoryStorageService, FboService& fboService, IWindow& window) :
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    window(window)
{
}

std::expected<Camera, std::string> CameraService::createCamera()
{
    auto windowState = window.state();

    auto camera = createCamera(CreateCameraDTO{.width = static_cast<unsigned int>(windowState.windowWidth),
                                               .height = static_cast<unsigned int>(windowState.windowHeight),
                                               .target = CameraTarget::ColourAndDepth});
    if (!camera)
    {
        return camera;
    }

    // The only difference between the screen camera and any other: this one's target is the
    // window's, so the window's size is the one it follows.
    camera->tracksWindowSize = true;

    return camera;
}

std::expected<Camera, std::string> CameraService::createCamera(const CreateCameraDTO& createCameraDTO)
{
    // A stated target first: a camera built over a composed framebuffer draws into attachments
    // other cameras own, so there is nothing here to create and the size fields are not read.
    auto output = createCameraDTO.output;
    if (output.has_value())
    {
        if (memoryStorageService.frameBuffers.find(output.value()) == nullptr)
        {
            return std::unexpected("the camera's stated render target names no live framebuffer");
        }
    }
    else
    {
        const auto wantsColour = createCameraDTO.target != CameraTarget::DepthOnly;
        const auto wantsDepth = createCameraDTO.target != CameraTarget::ColourOnly;

        std::vector<CreateFboAttachmentDTO> attachments;

        // Colour first, so the attachment order an existing camera's framebuffer has is unchanged
        // and so the presenter's "first colour attachment" stays the first element.
        if (wantsColour)
        {
            attachments.push_back(CreateFboAttachmentDTO{.width = createCameraDTO.width,
                                                         .height = createCameraDTO.height,
                                                         .type = FboAttachmentType::Color,
                                                         .captureFormat = TextureFormat::RGBA,
                                                         .internalFormat = TextureFormat::RGBA16F});
        }

        if (wantsDepth)
        {
            // A depth-only target is one a shader is going to compare against, so its precision is
            // named rather than left to the driver — see TextureFormat::DepthComponent32F. A camera
            // that also has colour is using its depth buffer as a depth *test*, where the driver's
            // choice has always been fine and changing it would move pixels for nothing.
            const auto depthFormat = wantsColour ? TextureFormat::DepthComponent : TextureFormat::DepthComponent32F;

            attachments.push_back(CreateFboAttachmentDTO{.width = createCameraDTO.width,
                                                         .height = createCameraDTO.height,
                                                         .type = FboAttachmentType::Depth,
                                                         .captureFormat = TextureFormat::DepthComponent,
                                                         .internalFormat = depthFormat,
                                                         .depthComparison = createCameraDTO.depthComparison});
        }

        // A camera is the render target it draws into; one that could not get its buffers is not a
        // camera with a missing field, so it is not handed back at all.
        const auto created =
            fboService.create(CreateFboDTO{.type = FboType::Planar, .attachments = std::move(attachments)});

        if (!created)
        {
            return std::unexpected("the camera has no output buffer: " + created.error());
        }

        output = created.value();
    }

    // The clipping planes default to the values the projection used to hardcode, so wiring the
    // fields changes no frame; a game that wants a different depth range now has one to set.
    // Exposure defaults to 1, meaning "hand the tone map the scene's own values". It was
    // value-initialised to zero for as long as nothing read it, which is a black frame the moment
    // something does — recorded in CLAUDE.md as a trap laid for whoever wired tone mapping. This
    // is that wiring, so the default moves with it.
    //
    // The shutter is derived rather than chosen: the film speed and the aperture were already here
    // and unread, and the exposure above is what the three of them together have to come out to.
    constexpr auto filmSpeed = 6400u;
    constexpr auto aperture = 1.4f;
    constexpr auto exposure = 1.0f;

    return Camera{.iso = filmSpeed,
                  .aspectRatio = 16.0f / 9.0f,
                  .aperture = aperture,
                  .shutterTime = shutterTimeForExposure(exposure, filmSpeed, aperture),
                  .exposure = exposure,
                  .fieldOfView = 75.f,
                  .role = createCameraDTO.role,
                  .debugName = createCameraDTO.debugName,
                  .overrideShader = createCameraDTO.overrideShader,
                  .tracksWindowSize = false,
                  .nearClippingPlane = 1.0f,
                  .farClippingPlane = 5000.0f,
                  .direction = glm::vec3(0, 0, 1),
                  .roll = glm::vec3(0, 1, 0),
                  .output = output};
}

void CameraService::setExposure(Camera& camera, const float exposure) const
{
    camera.exposure = exposure;
    camera.shutterTime = shutterTimeForExposure(exposure, camera.iso, camera.aperture);
    camera.autoExposure.adaptedLuminance = luminanceForExposure(exposure);
}

void CameraService::setFilmSpeed(Camera& camera, const unsigned int iso) const
{
    camera.iso = iso;
    camera.exposure = exposureFromTriangle(camera.iso, camera.aperture, camera.shutterTime);
}

void CameraService::setAperture(Camera& camera, const float aperture) const
{
    camera.aperture = aperture;
    camera.exposure = exposureFromTriangle(camera.iso, camera.aperture, camera.shutterTime);
}

void CameraService::setToneCurve(Camera& camera, const ToneCurve& toneCurve) const
{
    camera.toneCurve = toneCurve;
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

void CameraService::setPerspective(Camera& camera, float fov) const
{
    camera.projection = CameraProjection::Perspective;
    camera.fieldOfView = fov;
}

void CameraService::setOrthographic(Camera& camera, float left, float right, float bottom, float top) const
{
    camera.projection = CameraProjection::Orthographic;
    camera.orthographicVolume = OrthographicVolume{.left = left, .right = right, .bottom = bottom, .top = top};
}

const glm::mat4& CameraService::updateModelViewMatrix(Camera& camera) const
{
    camera.modelViewMatrix = glm::lookAt(camera.position, camera.position + camera.direction, camera.roll);
    return camera.modelViewMatrix;
}

glm::mat4 CameraService::projectionMatrix(const Camera& camera) const
{
    switch (camera.projection)
    {
    case CameraProjection::Orthographic:
        return glm::ortho(camera.orthographicVolume.left, camera.orthographicVolume.right,
                          camera.orthographicVolume.bottom, camera.orthographicVolume.top, camera.nearClippingPlane,
                          camera.farClippingPlane);
    case CameraProjection::Perspective:
        break;
    }

    return glm::perspective(glm::radians(camera.fieldOfView), camera.aspectRatio, camera.nearClippingPlane,
                            camera.farClippingPlane);
}

const glm::mat4& CameraService::updateModelViewProjectionMatrix(Camera& camera) const
{
    camera.modelViewProjectionMatrix = projectionMatrix(camera) * updateModelViewMatrix(camera);

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
