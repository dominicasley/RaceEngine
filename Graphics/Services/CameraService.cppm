module;

#include <expected>
#include <string>
#include <utility>
#include <vector>

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
    // The on-screen camera: colour and depth at the window's size, rebuilt when the window
    // resizes. Unchanged, and still the only call the game makes.
    [[nodiscard]] std::expected<Camera, std::string> createCamera();
    // Any other camera. The target is whatever the DTO asks for, at the resolution it asks for,
    // and a window resize leaves it alone — an off-screen camera is structurally just a camera
    // nothing presents, so a shadow cascade, a reflection probe and a post-FX pass all arrive
    // here.
    [[nodiscard]] std::expected<Camera, std::string> createCamera(const CreateCameraDTO& createCameraDTO);
    void setPosition(Camera& camera, float x, float y, float z) const;
    void setDirection(Camera& camera, float x, float y, float z) const;
    void translate(Camera& camera, float x, float y, float z) const;
    void rotate(Camera& camera, float x, float y, float z) const;
    void setRoll(Camera& camera, float x, float y, float z) const;
    void setAspectRatio(Camera& camera, float aspectRatio) const;
    void setClippingPlanes(Camera& camera, float nearPlane, float farPlane) const;
    [[nodiscard]] std::expected<void, std::string> recreateOutputBuffer(Camera& camera, int width, int height) const;
    void setFieldOfView(Camera& camera, float fov) const;
    // The two projection modes, each setting the mode and the parameters that mode reads. A
    // camera that never calls either is perspective with the field of view it was created with,
    // which is every camera that exists today.
    void setPerspective(Camera& camera, float fov) const;
    void setOrthographic(Camera& camera, float left, float right, float bottom, float top) const;
    void lookAtPoint(Camera& camera, float x, float y, float z) const;
    void addPostProcess(Camera& camera, const Resource<PostProcess>& postProcess) const;
    const Fbo& getOutputBuffer(Camera& camera) const;
    // View space to clip space, in the camera's own projection mode. Both backends keep the GL
    // depth convention here (z in -w..w); Vulkan applies its own 0..1 correction downstream.
    [[nodiscard]] glm::mat4 projectionMatrix(const Camera& camera) const;
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
    const auto wantsColour = createCameraDTO.target != CameraTarget::DepthOnly;
    const auto wantsDepth = createCameraDTO.target != CameraTarget::ColourOnly;

    std::vector<CreateFboAttachmentDTO> attachments;

    // Colour first, so the attachment order an existing camera's framebuffer has is unchanged and
    // so the presenter's "first colour attachment" stays the first element.
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
    const auto output = fboService.create(CreateFboDTO{.type = FboType::Planar, .attachments = std::move(attachments)});

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
                  .role = createCameraDTO.role,
                  .overrideShader = createCameraDTO.overrideShader,
                  .tracksWindowSize = false,
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

void CameraService::setPerspective(Camera& camera, float fov) const
{
    camera.projection = CameraProjection::Perspective;
    camera.fieldOfView = fov;
}

// Named left/right/bottom/top rather than width/height because a cascade's box is fitted to a
// frustum slice in light space and is not symmetric about the light's axis.
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

// The perspective arm is the expression this function used to be, unchanged and unconditional for
// a camera that never asked for anything else: the projection mode defaults to Perspective, so
// every existing camera produces the same matrix bit for bit.
//
// Both arms use the GL depth convention (glm's default, z in -w..w). Vulkan's 0..1 correction is
// applied by the Vulkan backend when it fills DrawData, and it is a post-multiply on the clip-space
// side, so it works the same for either projection (docs/vulkan-abi.md).
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
