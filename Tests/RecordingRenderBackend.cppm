module;

#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.tests.recording;

import raceengine;

namespace raceengine::tests
{

// One recording double for all three renderer seams, which is possible only because the seams
// take engine model types and nothing from an API: no VkDevice, no GL name, no window.
//
// Two jobs. It **records** what a service asked the backend for, so a test can assert on the
// call rather than on a pixel; and it **scripts failure**, because every creation on
// IGpuResourceFactory now reports std::expected and nothing else in the tree ever exercises the
// error arm — a driver that refuses a framebuffer is not something a gate can arrange.
//
// It implements the whole of IRenderBackend rather than one seam at a time so that a single
// instance can be handed to services that take different seams, the way the real backend is.
export class RecordingRenderBackend : public IRenderBackend
{
public:
    // Set any of these and the matching call reports it instead of succeeding.
    std::optional<std::string> shaderObjectFailure{};
    std::optional<std::string> cubeMapFailure{};
    std::optional<std::string> fboFailure{};
    std::optional<std::string> captureFailure{};
    std::optional<std::string> initFailure{};

    // False makes beginFrame report the swapchain-gone case: the step records nothing.
    bool frameOpens = true;

    struct FboRequest
    {
        FboType type;
        size_t attachments;
    };

    struct PresentRequest
    {
        Resource<Shader> shader;
        Resource<FboAttachment> attachment;
        // The lens and grade the presenter asked for, so a test can see that they reached the seam
        // rather than only that a present was recorded.
        glm::vec4 parameters{0.0f};
        bool graded = false;
    };

    struct CubeMapRequest
    {
        // Face names in the order IGpuResourceFactory declares them, so a test can prove the
        // service did not shuffle them on the way through.
        std::vector<std::string> faceNames;
    };

    std::vector<ShaderDescriptor> shaderObjectRequests;
    std::vector<CubeMapRequest> cubeMapRequests;
    std::vector<FboRequest> fboRequests;
    std::vector<std::optional<unsigned int>> deletedFboIds;
    std::vector<PresentRequest> presentRequests;
    std::vector<std::pair<GpuResourceKind, unsigned int>> releasedResources;
    std::vector<Resource<Material>> releasedMaterials;
    std::vector<std::string> captureRequests;
    unsigned int beginFrameCalls = 0;
    unsigned int endFrameCalls = 0;
    unsigned int recordViewCalls = 0;
    // Which probes the frame asked to capture, in order. A capture is a step rather than a whole
    // capture, so the same probe appears once per face it was advanced through — which is what a
    // test asserting the scheduler's pacing reads.
    std::vector<std::string> probeCaptureRequests;
    // Whether each recordAutoExposure call named a camera that meters, in call order.
    std::vector<bool> autoExposureRequests;
    // The same for the occlusion prepass: whether the camera it named gathers, in call order.
    std::vector<bool> ambientOcclusionRequests;
    std::optional<std::pair<int, int>> viewport{};

    // Every successful creation hands back a distinct id, so a test can tell one framebuffer's
    // id from another's and can see a recreate produce a new one.
    unsigned int nextGpuResourceId = 1;

    [[nodiscard]] std::expected<void, std::string> init() override
    {
        if (initFailure.has_value())
        {
            return std::unexpected(initFailure.value());
        }

        return {};
    }

    void setViewport(const int width, const int height) override
    {
        viewport = std::pair{width, height};
    }

    [[nodiscard]] bool beginFrame() override
    {
        beginFrameCalls++;

        return frameOpens;
    }

    void recordView(Scene&, Camera&, float) override
    {
        recordViewCalls++;
    }

    // The fake captures nothing, so it advances the probe the way a backend would: one face per
    // call, and Ready once the sixth has been asked for. Without that the scheduler would hand it
    // the same probe every frame forever and no test could see a second one.
    void recordProbeCapture(Scene&, LightProbe& probe) override
    {
        probeCaptureRequests.push_back(probe.name);

        probe.captureFace++;
        probe.state = probe.captureFace >= 6 ? LightProbeState::Ready : LightProbeState::Capturing;
    }

    // The fake draws nothing, so no occlusion is gathered. What it records is that the frame asked,
    // and for a camera that has anything to gather — the ordering against recordView is the half of
    // this seam a test without a device can see.
    void recordAmbientOcclusion(Scene&, Camera& camera, float) override
    {
        ambientOcclusionRequests.push_back(camera.ambientOcclusion.enabled);
    }

    // The fake takes no reading, so a metered camera's exposure never moves here. What it records
    // is that the frame asked at all, and for which camera — the ordering against recordView is
    // what a test of the step's shape reads.
    void recordAutoExposure(Camera& camera) override
    {
        autoExposureRequests.push_back(camera.autoExposure.enabled);
    }

    void recordPresent(const Presenter& presenter) override
    {
        presentRequests.push_back(PresentRequest{.shader = presenter.shader,
                                                 .attachment = presenter.output,
                                                 .parameters = presenter.parameters,
                                                 .graded = presenter.lookupTable.has_value()});
    }

    void endFrame() override
    {
        endFrameCalls++;
    }

    [[nodiscard]] std::expected<unsigned int, std::string>
    createShaderObject(const ShaderDescriptor& shaderDescriptor) override
    {
        shaderObjectRequests.push_back(shaderDescriptor);

        if (shaderObjectFailure.has_value())
        {
            return std::unexpected(shaderObjectFailure.value());
        }

        return nextGpuResourceId++;
    }

    [[nodiscard]] std::expected<unsigned int, std::string> createCubeMap(const Texture& front, const Texture& back,
                                                                         const Texture& left, const Texture& right,
                                                                         const Texture& top,
                                                                         const Texture& bottom) override
    {
        cubeMapRequests.push_back(
            CubeMapRequest{.faceNames = {front.name, back.name, left.name, right.name, top.name, bottom.name}});

        if (cubeMapFailure.has_value())
        {
            return std::unexpected(cubeMapFailure.value());
        }

        return nextGpuResourceId++;
    }

    [[nodiscard]] std::expected<unsigned int, std::string> createFbo(const Fbo& fbo) override
    {
        fboRequests.push_back(FboRequest{.type = fbo.type, .attachments = fbo.attachments.size()});

        if (fboFailure.has_value())
        {
            return std::unexpected(fboFailure.value());
        }

        return nextGpuResourceId++;
    }

    void deleteFbo(Fbo& fbo) override
    {
        deletedFboIds.push_back(fbo.gpuResourceId);
        fbo.gpuResourceId.reset();
    }

    void releaseGpuResource(const GpuResourceKind kind, const unsigned int gpuResourceId) override
    {
        releasedResources.emplace_back(kind, gpuResourceId);
    }

    void releaseMaterial(const Resource<Material>& material) override
    {
        releasedMaterials.push_back(material);
    }

    [[nodiscard]] GpuResourceCensus gpuResourceCensus() const override
    {
        // Created minus released, counted as one pool: the tests here care whether a release
        // reached the backend at all, not which family it landed in.
        const auto created = nextGpuResourceId - 1;

        return GpuResourceCensus{.textures = created - static_cast<unsigned int>(releasedResources.size())};
    }

    [[nodiscard]] std::expected<void, std::string> captureFrame(const std::string& path) override
    {
        captureRequests.push_back(path);

        if (captureFailure.has_value())
        {
            return std::unexpected(captureFailure.value());
        }

        return {};
    }
};

} // namespace raceengine::tests
