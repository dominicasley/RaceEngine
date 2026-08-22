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

// Film speed and aperture hold the shutter and let the picture move, which is the way round a
// camera works: opening up two stops is two stops brighter, and it is the metered case that then
// puts the shutter back to hold the picture instead.

// The other half of the same decision, and separable from it: exposure says how much light the
// frame has, this says what the curve does with it. The default is already the dramatic reading
// (see ToneCurve), so a game calls this to soften the picture rather than to get one.

// x, y and z are a delta applied to the current direction, not a direction: there is no
// predicate over them that says the camera would not move, so this writes unconditionally.

// Named nearPlane/farPlane, not near/far: those are still macros in the Windows headers the
// window layer pulls in.

// Named left/right/bottom/top rather than width/height because a cascade's box is fitted to a
// frustum slice in light space and is not symmetric about the light's axis.

// The perspective arm is the expression this function used to be, unchanged and unconditional for
// a camera that never asked for anything else: the projection mode defaults to Perspective, so
// every existing camera produces the same matrix bit for bit.
//
// Both arms use the GL depth convention (glm's default, z in -w..w). Vulkan's 0..1 correction is
// applied by the Vulkan backend when it fills DrawData, and it is a post-multiply on the clip-space
// side, so it works the same for either projection (docs/vulkan-abi.md).

} // namespace raceengine
