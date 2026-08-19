module;

#include <expected>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module raceengine.graphics:CameraService;

import :FboService;
import :PhysicalCamera;
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
    void setExposure(Camera& camera, float exposure) const;
    // The other two legs of the exposure triangle. Both hold the shutter and move the picture, as
    // they do on a camera: two stops between f/1.4 and f/2.8, one per doubling of the film speed.
    void setFilmSpeed(Camera& camera, unsigned int iso) const;
    void setAperture(Camera& camera, float aperture) const;
    void setToneCurve(Camera& camera, const ToneCurve& toneCurve) const;
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
                  .overrideShader = createCameraDTO.overrideShader,
                  .tracksWindowSize = false,
                  .nearClippingPlane = 1.0f,
                  .farClippingPlane = 5000.0f,
                  .direction = glm::vec3(0, 0, 1),
                  .roll = glm::vec3(0, 1, 0),
                  .output = output.value()};
}

// A linear multiplier applied to the scene's radiance before the tone map, which is what an
// exposure is: the renderer works in relative units, so the number that turns "sunlight is 3.2"
// into a picture is the game's to state.
//
// It states it through the shutter rather than beside it. The multiplier is a function of all three
// legs of the triangle (see PhysicalCamera), so writing it alone would leave a camera whose film
// speed and aperture disagreed with its own picture — and auto exposure, which moves the shutter,
// would then be adjusting a leg nothing was reading. Back-solving keeps one source of truth and
// makes this call exactly what it reads as on a camera: the setting the meter starts from and, with
// no meter, holds.
void CameraService::setExposure(Camera& camera, const float exposure) const
{
    camera.exposure = exposure;
    camera.shutterTime = shutterTimeForExposure(exposure, camera.iso, camera.aperture);
    camera.autoExposure.adaptedLuminance = luminanceForExposure(exposure);
}

// Film speed and aperture hold the shutter and let the picture move, which is the way round a
// camera works: opening up two stops is two stops brighter, and it is the metered case that then
// puts the shutter back to hold the picture instead.
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

// The other half of the same decision, and separable from it: exposure says how much light the
// frame has, this says what the curve does with it. The default is already the dramatic reading
// (see ToneCurve), so a game calls this to soften the picture rather than to get one.
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
