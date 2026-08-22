// AutoExposureService bodies. Declarations are in Graphics/Services/AutoExposureService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <cmath>
#include <expected>
#include <string>

#include <glm/glm.hpp>
#include <spdlog/logger.h>

module raceengine.graphics;

import :AutoExposureService;
import :CameraService;
import :FboService;
import :PhysicalCamera;
import :PostProcessing;
import :PostProcessService;
import :RenderContract;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

AutoExposureService::AutoExposureService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                                         FboService& fboService, PostProcessService& postProcessService,
                                         CameraService& cameraService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    fboService(fboService),
    postProcessService(postProcessService),
    cameraService(cameraService)
{
}

std::expected<void, std::string> AutoExposureService::enable(Camera& camera, const CreateAutoExposureDTO& dto) const
{
    if (!camera.output.has_value())
    {
        return std::unexpected("a camera with no render target has no frame to meter");
    }

    const auto* cameraBuffer = memoryStorageService.frameBuffers.find(camera.output.value());
    if (cameraBuffer == nullptr)
    {
        return std::unexpected("the camera's render target handle names no live framebuffer");
    }

    // Resolved before the reduction buffer is created rather than after: this is a borrow out of
    // the same storage the creation below adds to.
    const auto sceneColour = fboService.getAttachmentsOfType(*cameraBuffer, FboAttachmentType::Color);
    if (sceneColour.empty())
    {
        return std::unexpected("the camera's render target has no colour attachment to meter");
    }

    const auto levels = mipLevelCount(exposureMeterResolution, exposureMeterResolution);

    const auto reduction =
        fboService.create(CreateFboDTO{.type = FboType::Planar,
                                       .attachments = {CreateFboAttachmentDTO{.width = exposureMeterResolution,
                                                                              .height = exposureMeterResolution,
                                                                              .type = FboAttachmentType::Color,
                                                                              .captureFormat = TextureFormat::RGBA,
                                                                              .internalFormat = TextureFormat::RGBA16F,
                                                                              .levels = levels}}});
    if (!reduction)
    {
        return std::unexpected("the exposure meter has no reduction buffer: " + reduction.error());
    }

    const auto chain = fboService.getAttachmentsOfType(memoryStorageService.frameBuffers.get(reduction.value()),
                                                       FboAttachmentType::Color);
    if (chain.empty())
    {
        return std::unexpected("the exposure meter's reduction buffer has no colour attachment");
    }

    for (auto level = 0u; level < levels; level++)
    {
        const auto pass = postProcessService.create(dto.shader, reduction.value(), level);

        // Level zero reads the frame; every level below reads the one above it out of the image it
        // is itself writing a level of, which is what FboAttachment::levels and the backend's
        // per-level layout tracking exist for.
        postProcessService.addInput(pass, level == 0 ? sceneColour.front() : chain.front(),
                                    level == 0 ? 0u : level - 1u);
        // Only the top of the chain weights anything: it is the one pass that sees the frame, and
        // every level below it is averaging numbers that already carry their weight with them.
        postProcessService.setParameters(pass, glm::vec4(dto.meter.centreWeighting, 0.0f, 0.0f, 0.0f));
        cameraService.addPostProcess(camera, pass);
    }

    camera.autoExposure = dto.meter;
    camera.autoExposure.enabled = true;
    camera.autoExposure.chain = chain.front();
    camera.autoExposure.chainLevel = levels - 1;
    camera.autoExposure.measuredLuminance = 0.0f;
    camera.autoExposure.settleLogged = false;
    // Seeded from whatever the level set by hand, so the manual exposure is where the adaptation
    // starts rather than a value it discards on the first reading.
    camera.autoExposure.adaptedLuminance = luminanceForExposure(camera.exposure);

    logger.info("Auto exposure enabled: {} reduction level(s) from {}x{} to one texel, compensation {:+.2f} EV, "
                "centre weighting {:.2f}, starting from exposure {:.2f}",
                levels, exposureMeterResolution, exposureMeterResolution, camera.autoExposure.compensation,
                camera.autoExposure.centreWeighting, camera.exposure);

    return {};
}

void AutoExposureService::update(Camera& camera, const unsigned int ticks, const float tickSeconds) const
{
    auto& meter = camera.autoExposure;

    // Nothing has come back yet — the first reading is three submissions behind the first frame
    // that recorded a copy — so the camera holds the exposure the level gave it. That is the
    // starting point doing its job, not a missing value being papered over.
    if (!meter.enabled || meter.measuredLuminance <= 0.0f)
    {
        return;
    }

    const auto speed =
        meter.measuredLuminance > meter.adaptedLuminance ? meter.lightAdaptationSpeed : meter.darkAdaptationSpeed;
    meter.adaptedLuminance =
        adaptLuminance(meter.adaptedLuminance, meter.measuredLuminance, speed, static_cast<float>(ticks) * tickSeconds);

    const auto setting = meterExposure(meter.adaptedLuminance, meter, camera.iso, camera.aperture);
    camera.shutterTime = setting.shutterTime;
    camera.exposure = setting.exposure;

    // Stated once, when the adaptation has closed to within a hundredth of a stop of the reading:
    // what a level author wants to know is what the scene meters at, not the sequence of values the
    // camera passed through on the way to it. A camera that never settles never says so.
    if (!meter.settleLogged && std::abs(std::log2(meter.adaptedLuminance / meter.measuredLuminance)) < 0.01f)
    {
        logger.info("Auto exposure settled: scene luminance {:.4f}, EV100 {:.2f}, shutter 1/{:.0f} s at f/{:.1f} and "
                    "ISO {}, exposure {:.2f}",
                    meter.measuredLuminance, setting.ev100, 1.0f / setting.shutterTime, camera.aperture, camera.iso,
                    setting.exposure);
        meter.settleLogged = true;
    }
}

} // namespace raceengine
