#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

import raceengine;
import raceengine.tests.recording;

using raceengine::Camera;
using raceengine::CameraProjection;
using raceengine::CameraService;
using raceengine::CameraTarget;
using raceengine::CreateCameraDTO;
using raceengine::CursorMode;
using raceengine::DepthComparison;
using raceengine::FboAttachmentType;
using raceengine::FboService;
using raceengine::FboType;
using raceengine::IWindow;
using raceengine::Key;
using raceengine::MemoryStorageService;
using raceengine::TextureFormat;
using raceengine::WindowState;
using raceengine::tests::RecordingRenderBackend;

namespace
{

// A window that is only its size. CameraService reads nothing else, and a camera's render target
// is the one thing the window's size decides — which is exactly what the off-screen cases below
// have to stop depending on.
class FixedSizeWindow : public IWindow
{
public:
    explicit FixedSizeWindow(const int width, const int height) :
        windowState(WindowState{.mouseX = 0.0, .mouseY = 0.0, .windowWidth = width, .windowHeight = height})
    {
    }

    void makeContextCurrent() override
    {
    }

    void swapBuffers() const override
    {
    }

    void setMousePosition(int, int) override
    {
    }

    void setCursorMode(CursorMode) override
    {
    }

    [[nodiscard]] std::tuple<double, double> mouseDelta() override
    {
        return {0.0, 0.0};
    }

    [[nodiscard]] bool shouldClose() const override
    {
        return false;
    }

    [[nodiscard]] bool keyPressed(Key) const override
    {
        return false;
    }

    [[nodiscard]] const WindowState& state() const override
    {
        return windowState;
    }

    [[nodiscard]] std::tuple<double, double> mousePosition() override
    {
        return {0.0, 0.0};
    }

    [[nodiscard]] float delta() const override
    {
        return 0.0f;
    }

private:
    WindowState windowState;
};

struct CameraFixture
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    FixedSizeWindow window{1920, 1080};
    FboService fboService{storage, backend};
    CameraService cameraService{storage, fboService, window};

    // The attachments the camera's framebuffer actually carries, in the order the backend was
    // handed them.
    [[nodiscard]] std::vector<raceengine::FboAttachment> attachmentsOf(Camera& camera)
    {
        std::vector<raceengine::FboAttachment> attachments;

        for (const auto& attachmentKey : cameraService.getOutputBuffer(camera).attachments)
        {
            const auto* attachment = storage.bufferAttachments.find(attachmentKey);
            REQUIRE(attachment != nullptr);
            attachments.push_back(*attachment);
        }

        return attachments;
    }
};

} // namespace

TEST_CASE("createCamera with no argument is still colour and depth at the window's size", "[camera][fbo]")
{
    CameraFixture fixture;

    auto camera = fixture.cameraService.createCamera();
    REQUIRE(camera.has_value());

    const auto attachments = fixture.attachmentsOf(camera.value());
    REQUIRE(attachments.size() == 2);
    REQUIRE(attachments[0].type == FboAttachmentType::Color);
    REQUIRE(attachments[0].internalFormat == TextureFormat::RGBA16F);
    REQUIRE(attachments[1].type == FboAttachmentType::Depth);

    for (const auto& attachment : attachments)
    {
        REQUIRE(attachment.width == 1920);
        REQUIRE(attachment.height == 1080);
        // Off unless asked for: a comparison sampler bound where a shader declares a plain
        // sampler is undefined, so the screen camera must not acquire one by default.
        REQUIRE(attachment.depthComparison == DepthComparison::None);
    }

    // The screen camera follows the window, which is what makes the resize path rebuild it.
    REQUIRE(camera->tracksWindowSize);
    REQUIRE(camera->projection == CameraProjection::Perspective);

    REQUIRE(fixture.backend.fboRequests.size() == 1);
    REQUIRE(fixture.backend.fboRequests.front().type == FboType::Planar);
    REQUIRE(fixture.backend.fboRequests.front().attachments == 2);
}

TEST_CASE("a depth-only camera gets one depth attachment at its own resolution", "[camera][fbo][depth]")
{
    CameraFixture fixture;

    auto camera = fixture.cameraService.createCamera(CreateCameraDTO{.width = 2048,
                                                                     .height = 2048,
                                                                     .target = CameraTarget::DepthOnly,
                                                                     .depthComparison = DepthComparison::LessOrEqual});
    REQUIRE(camera.has_value());

    const auto attachments = fixture.attachmentsOf(camera.value());
    REQUIRE(attachments.size() == 1);
    REQUIRE(attachments.front().type == FboAttachmentType::Depth);
    REQUIRE(attachments.front().internalFormat == TextureFormat::DepthComponent);

    // Sized independently of the window, which is the whole point: the window here is 1920x1080.
    REQUIRE(attachments.front().width == 2048);
    REQUIRE(attachments.front().height == 2048);

    // What both backends read to build a comparison sampler — GL_TEXTURE_COMPARE_MODE on the
    // texture object, compareEnable on the Vulkan sampler.
    REQUIRE(attachments.front().depthComparison == DepthComparison::LessOrEqual);

    // Nothing rebuilds it when the window resizes.
    REQUIRE_FALSE(camera->tracksWindowSize);

    // The backend was handed a framebuffer with exactly one attachment: a colour-less target is
    // what GL_NONE draw/read buffers and a zero-colour VkRenderingInfo are for.
    REQUIRE(fixture.backend.fboRequests.size() == 1);
    REQUIRE(fixture.backend.fboRequests.front().attachments == 1);
}

TEST_CASE("a colour-only camera gets no depth attachment", "[camera][fbo]")
{
    CameraFixture fixture;

    auto camera = fixture.cameraService.createCamera(
        CreateCameraDTO{.width = 512, .height = 256, .target = CameraTarget::ColourOnly});
    REQUIRE(camera.has_value());

    const auto attachments = fixture.attachmentsOf(camera.value());
    REQUIRE(attachments.size() == 1);
    REQUIRE(attachments.front().type == FboAttachmentType::Color);
    REQUIRE(attachments.front().width == 512);
    REQUIRE(attachments.front().height == 256);
}

TEST_CASE("an off-screen camera the backend refused is not handed back", "[camera][fbo][expected]")
{
    CameraFixture fixture;
    fixture.backend.fboFailure = "out of device memory";

    const auto camera = fixture.cameraService.createCamera(
        CreateCameraDTO{.width = 2048, .height = 2048, .target = CameraTarget::DepthOnly});

    REQUIRE_FALSE(camera.has_value());
    REQUIRE(camera.error() == "the camera has no output buffer: out of device memory");
}

TEST_CASE("a perspective camera's projection is the matrix it has always been", "[camera][projection]")
{
    CameraFixture fixture;

    auto camera = fixture.cameraService.createCamera();
    REQUIRE(camera.has_value());

    fixture.cameraService.setPosition(camera.value(), 0.0f, 600.0f, -450.0f);
    fixture.cameraService.setRoll(camera.value(), 0.0f, 1.0f, 0.0f);
    fixture.cameraService.lookAtPoint(camera.value(), 0.0f, 0.0f, 0.0f);

    // Bit-identical to the expression updateModelViewProjectionMatrix used to hardcode. Written
    // out here rather than compared approximately: the parity gate demands the frame not move by
    // a pixel, and a projection that is merely close would move it.
    const auto expected = glm::perspective(glm::radians(camera->fieldOfView), camera->aspectRatio,
                                           camera->nearClippingPlane, camera->farClippingPlane);

    REQUIRE(fixture.cameraService.projectionMatrix(camera.value()) == expected);

    const auto view = fixture.cameraService.updateModelViewMatrix(camera.value());
    REQUIRE(fixture.cameraService.updateModelViewProjectionMatrix(camera.value()) == expected * view);
}

TEST_CASE("an orthographic camera projects its own volume", "[camera][projection]")
{
    CameraFixture fixture;

    auto camera = fixture.cameraService.createCamera(
        CreateCameraDTO{.width = 2048, .height = 2048, .target = CameraTarget::DepthOnly});
    REQUIRE(camera.has_value());

    fixture.cameraService.setClippingPlanes(camera.value(), 1.0f, 1000.0f);
    fixture.cameraService.setOrthographic(camera.value(), -300.0f, 500.0f, -200.0f, 400.0f);

    REQUIRE(camera->projection == CameraProjection::Orthographic);

    // The near and far planes are Camera's own, so the two modes state their depth range the same
    // way; only the side planes come from the volume. Asymmetric on purpose: a cascade's box is
    // fitted to a frustum slice and is not centred on the light's axis.
    const auto expected = glm::ortho(-300.0f, 500.0f, -200.0f, 400.0f, 1.0f, 1000.0f);
    REQUIRE(fixture.cameraService.projectionMatrix(camera.value()) == expected);

    // GL's depth convention, which is what the whole engine keeps: a point on the near plane
    // lands at NDC z -1 and one on the far plane at +1. Vulkan's 0..1 correction is applied by
    // the Vulkan backend, downstream of here.
    const auto onNear = expected * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);
    const auto onFar = expected * glm::vec4(0.0f, 0.0f, -1000.0f, 1.0f);
    REQUIRE(onNear.z / onNear.w == -1.0f);
    REQUIRE(onFar.z / onFar.w == 1.0f);
}

TEST_CASE("switching projection mode keeps the other mode's framing", "[camera][projection]")
{
    CameraFixture fixture;

    auto camera = fixture.cameraService.createCamera();
    REQUIRE(camera.has_value());

    const auto perspective = fixture.cameraService.projectionMatrix(camera.value());

    fixture.cameraService.setOrthographic(camera.value(), -1.0f, 1.0f, -1.0f, 1.0f);
    REQUIRE(fixture.cameraService.projectionMatrix(camera.value()) != perspective);

    // setPerspective restores the mode, and the field of view and aspect ratio were never
    // overwritten by the orthographic volume — the two sets of parameters coexist.
    fixture.cameraService.setPerspective(camera.value(), camera->fieldOfView);
    REQUIRE(fixture.cameraService.projectionMatrix(camera.value()) == perspective);
}
