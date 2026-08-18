module;

#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

    void recordPresent(const Resource<Shader>& shader, const Resource<FboAttachment>& attachment) override
    {
        presentRequests.push_back(PresentRequest{.shader = shader, .attachment = attachment});
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
