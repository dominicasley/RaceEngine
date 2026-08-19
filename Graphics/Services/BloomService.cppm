module;

#include <algorithm>
#include <expected>
#include <string>

#include <glm/glm.hpp>
#include <spdlog/logger.h>

export module raceengine.graphics:BloomService;

import :CameraService;
import :FboService;
import :PostProcessing;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class BloomService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    FboService& fboService;
    PostProcessService& postProcessService;
    CameraService& cameraService;

public:
    explicit BloomService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, FboService& fboService,
                          PostProcessService& postProcessService, CameraService& cameraService);
    // Builds the two chains and every pass over them, and appends the passes to the camera's own
    // list in the order they have to run. Enable it *before* the pass that tone maps: the spill is
    // added to the frame in the exposed domain, so what reads it has to come after what writes it.
    [[nodiscard]] std::expected<void, std::string> enable(Camera& camera, const CreateBloomDTO& dto) const;
    // Both chains at the window's new size. Half of it, as they were built.
    [[nodiscard]] std::expected<void, std::string> resize(const Camera& camera, int width, int height) const;
};

BloomService::BloomService(spdlog::logger& logger, MemoryStorageService& memoryStorageService, FboService& fboService,
                           PostProcessService& postProcessService, CameraService& cameraService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    postProcessService(postProcessService),
    cameraService(cameraService)
{
}

std::expected<void, std::string> BloomService::enable(Camera& camera, const CreateBloomDTO& dto) const
{
    if (!camera.output.has_value())
    {
        return std::unexpected("a camera with no render target has nothing to bloom");
    }

    const auto* cameraBuffer = memoryStorageService.frameBuffers.find(camera.output.value());
    if (cameraBuffer == nullptr)
    {
        return std::unexpected("the camera's render target handle names no live framebuffer");
    }

    const auto sceneColour = fboService.getAttachmentsOfType(*cameraBuffer, FboAttachmentType::Color);
    if (sceneColour.empty())
    {
        return std::unexpected("the camera's render target has no colour attachment to threshold");
    }

    const auto* sizeSource = memoryStorageService.bufferAttachments.find(sceneColour.front());
    if (sizeSource == nullptr)
    {
        return std::unexpected("the camera's colour attachment is no longer loaded");
    }

    // Half the view, which is where a bloom chain starts: the first halving is free quality — the
    // spill is a wide blur and nothing in it survives at a pixel — and every level below is a
    // quarter of the work it would otherwise be.
    const auto width = std::max(sizeSource->width / 2u, 1u);
    const auto height = std::max(sizeSource->height / 2u, 1u);
    const auto levels = std::min(mipLevelCount(width, height), bloomMaxLevels);

    if (levels < 2)
    {
        return std::unexpected("a view this small has no chain to bloom down");
    }

    const auto chain = [&](const char* what) -> std::expected<Resource<Fbo>, std::string>
    {
        auto created =
            fboService.create(CreateFboDTO{.type = FboType::Planar,
                                           .attachments = {CreateFboAttachmentDTO{.width = width,
                                                                                  .height = height,
                                                                                  .type = FboAttachmentType::Color,
                                                                                  .captureFormat = TextureFormat::RGBA,
                                                                                  .internalFormat = TextureFormat::RGBA16F,
                                                                                  .levels = levels}}});
        if (!created)
        {
            return std::unexpected(std::string("the bloom ") + what + " chain has no buffer: " + created.error());
        }

        return created.value();
    };

    const auto downsample = chain("downsample");
    if (!downsample)
    {
        return std::unexpected(downsample.error());
    }

    const auto upsample = chain("upsample");
    if (!upsample)
    {
        return std::unexpected(upsample.error());
    }

    const auto downsampleColour = fboService.getAttachmentsOfType(
        memoryStorageService.frameBuffers.get(downsample.value()), FboAttachmentType::Color);
    const auto upsampleColour = fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(upsample.value()),
                                                                FboAttachmentType::Color);
    if (downsampleColour.empty() || upsampleColour.empty())
    {
        return std::unexpected("a bloom chain came back without a colour attachment");
    }

    std::vector<Resource<PostProcess>> passes;

    // Down: level zero reads the frame and keeps what is over the threshold; every level below
    // reads the one above it out of the chain it is itself writing a level of.
    for (auto level = 0u; level < levels; level++)
    {
        const auto pass = postProcessService.create(dto.downsampleShader, downsample.value(), level);
        postProcessService.addInput(pass, level == 0 ? sceneColour.front() : downsampleColour.front(),
                                    level == 0 ? 0u : level - 1u);
        postProcessService.setParameters(pass,
                                        glm::vec4(dto.bloom.threshold, dto.bloom.knee, dto.bloom.maximum, 0.0f));
        cameraService.addPostProcess(camera, pass);
        passes.push_back(pass);
    }

    // Up: the deepest level of the upsample chain is the deepest level of the downsample chain, so
    // the walk starts one above it and there is no pass that only copies. Each one reads the level
    // below out of the upsample chain and the matching level out of the downsample chain.
    for (auto level = levels - 1; level > 0; level--)
    {
        const auto target = level - 1;
        const auto pass = postProcessService.create(dto.upsampleShader, upsample.value(), target);
        postProcessService.addInput(pass, level == levels - 1 ? downsampleColour.front() : upsampleColour.front(),
                                    level);
        postProcessService.addInput(pass, downsampleColour.front(), target);
        postProcessService.setParameters(pass, glm::vec4(dto.bloom.spread, 0.0f, 0.0f, 0.0f));
        cameraService.addPostProcess(camera, pass);
        passes.push_back(pass);
    }

    camera.bloom = dto.bloom;
    camera.bloom.enabled = true;
    camera.bloom.downsample = downsample.value();
    camera.bloom.upsample = upsample.value();
    camera.bloom.result = upsampleColour.front();
    camera.bloom.passes = std::move(passes);

    logger.info("Bloom enabled: {} level(s) from {}x{}, threshold {:.2f} with a {:.2f} knee, clamped at {:.0f}, "
                "intensity {:.3f}",
                levels, width, height, camera.bloom.threshold, camera.bloom.knee, camera.bloom.maximum,
                camera.bloom.intensity);

    return {};
}

std::expected<void, std::string> BloomService::resize(const Camera& camera, const int width, const int height) const
{
    if (!camera.bloom.enabled || !camera.bloom.downsample.has_value() || !camera.bloom.upsample.has_value())
    {
        return {};
    }

    const auto halfWidth = std::max(static_cast<unsigned int>(std::max(width, 1)) / 2u, 1u);
    const auto halfHeight = std::max(static_cast<unsigned int>(std::max(height, 1)) / 2u, 1u);

    for (const auto& buffer : {camera.bloom.downsample.value(), camera.bloom.upsample.value()})
    {
        if (const auto resized = fboService.resize(buffer, halfWidth, halfHeight); !resized)
        {
            return std::unexpected("a bloom chain was not rebuilt: " + resized.error());
        }
    }

    // The chain's length is a function of its size, and a rebuilt buffer keeps the level count it
    // was created with — so a window resized far enough would leave passes writing levels that no
    // longer exist. The backend clamps a level past the end onto the last one, which is a blurrier
    // bloom rather than a broken frame; anything better means rebuilding the passes, and a resize
    // is not the place to be reallocating a camera's chain.
    if (mipLevelCount(halfWidth, halfHeight) < camera.bloom.passes.size() / 2)
    {
        logger.warn("Bloom chain is longer than {}x{} supports; the deepest levels now share the last one", halfWidth,
                    halfHeight);
    }

    return {};
}

} // namespace raceengine
