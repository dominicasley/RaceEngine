#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

import raceengine;
import raceengine.tests.recording;

using raceengine::CreateFboAttachmentDTO;
using raceengine::CreateFboDTO;
using raceengine::CubeMapService;
using raceengine::FboAttachmentType;
using raceengine::FboService;
using raceengine::FboType;
using raceengine::MemoryStorageService;
using raceengine::PixelDataType;
using raceengine::Presenter;
using raceengine::PresenterService;
using raceengine::Resource;
using raceengine::Shader;
using raceengine::ShaderDescriptor;
using raceengine::ShaderService;
using raceengine::Texture;
using raceengine::TextureFormat;
using raceengine::tests::RecordingRenderBackend;

namespace
{

CreateFboDTO colourAndDepth(const unsigned int width, const unsigned int height)
{
    return CreateFboDTO{.type = FboType::Planar,
                        .attachments = {CreateFboAttachmentDTO{.width = width,
                                                               .height = height,
                                                               .type = FboAttachmentType::Color,
                                                               .captureFormat = TextureFormat::RGBA,
                                                               .internalFormat = TextureFormat::RGBA16F},
                                        CreateFboAttachmentDTO{.width = width,
                                                               .height = height,
                                                               .type = FboAttachmentType::Depth,
                                                               .captureFormat = TextureFormat::DepthComponent,
                                                               .internalFormat = TextureFormat::DepthComponent}}};
}

Texture face(const std::string& name)
{
    return Texture{.name = name,
                   .format = TextureFormat::RGB,
                   .pixelDataType = PixelDataType::UnsignedByte,
                   .width = 1,
                   .height = 1,
                   .data = std::vector<unsigned char>{1, 2, 3}};
}

} // namespace

TEST_CASE("FboService::create writes the backend's id through the stored framebuffer", "[fbo][seam]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const FboService fboService(storage, backend);

    const auto created = fboService.create(colourAndDepth(320, 240));

    REQUIRE(created.has_value());
    REQUIRE(storage.frameBuffers.exists(created.value()));

    const auto* stored = storage.frameBuffers.find(created.value());
    REQUIRE(stored != nullptr);
    REQUIRE(stored->gpuResourceId.has_value());
    REQUIRE(stored->attachments.size() == 2);

    // The backend was handed the framebuffer with its attachments already in storage, which is
    // what lets a real backend size its images from them.
    REQUIRE(backend.fboRequests.size() == 1);
    REQUIRE(backend.fboRequests.front().attachments == 2);
    REQUIRE(backend.fboRequests.front().type == FboType::Planar);

    const auto* attachment = storage.bufferAttachments.find(stored->attachments.front());
    REQUIRE(attachment != nullptr);
    REQUIRE(attachment->width == 320);
    REQUIRE(attachment->height == 240);
}

TEST_CASE("FboService::create reports what the backend reported", "[fbo][seam][expected]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    backend.fboFailure = "framebuffer incomplete: attachment";
    const FboService fboService(storage, backend);

    const auto created = fboService.create(colourAndDepth(320, 240));

    REQUIRE_FALSE(created.has_value());
    REQUIRE(created.error() == "framebuffer incomplete: attachment");

    // Characterization: the attachments are added to storage before the framebuffer is asked
    // for, so a refused framebuffer leaves its attachment elements behind. Nothing removes them
    // — the handles that named them were never returned to a caller.
    REQUIRE(storage.bufferAttachments.size() == 2);
    REQUIRE(storage.frameBuffers.size() == 0);
}

TEST_CASE("FboService::recreate rejects a handle that names nothing", "[fbo][seam][expected]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const FboService fboService(storage, backend);

    const auto created = fboService.create(colourAndDepth(320, 240));
    REQUIRE(created.has_value());
    REQUIRE(storage.frameBuffers.remove(created.value()));

    const auto recreated = fboService.recreate(created.value());

    REQUIRE_FALSE(recreated.has_value());
    REQUIRE(recreated.error() == "the framebuffer handle names no live framebuffer");
    REQUIRE(backend.deletedFboIds.empty());
}

TEST_CASE("FboService::recreate replaces the id through the same handle", "[fbo][seam]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const FboService fboService(storage, backend);

    const auto created = fboService.create(colourAndDepth(320, 240));
    REQUIRE(created.has_value());

    const auto idBefore = storage.frameBuffers.find(created.value())->gpuResourceId;
    REQUIRE(idBefore.has_value());

    REQUIRE(fboService.recreate(created.value()).has_value());

    REQUIRE(backend.deletedFboIds.size() == 1);
    REQUIRE(backend.deletedFboIds.front() == idBefore);

    const auto idAfter = storage.frameBuffers.find(created.value())->gpuResourceId;
    REQUIRE(idAfter.has_value());
    REQUIRE(idAfter.value() != idBefore.value());
}

TEST_CASE("a framebuffer that fails to come back is left with no id", "[fbo][seam][expected]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const FboService fboService(storage, backend);

    const auto created = fboService.create(colourAndDepth(320, 240));
    REQUIRE(created.has_value());
    REQUIRE(storage.frameBuffers.find(created.value())->gpuResourceId.has_value());

    backend.fboFailure = "out of device memory";

    const auto recreated = fboService.recreate(created.value());

    REQUIRE_FALSE(recreated.has_value());
    REQUIRE(recreated.error() == "out of device memory");

    // The stale id was invalidated by the delete that preceded the failed create. Leaving it in
    // place would hand whoever handles the error a framebuffer that can still be drawn into by
    // accident; this is the one thing the caller must be able to rely on after a failure.
    REQUIRE_FALSE(storage.frameBuffers.find(created.value())->gpuResourceId.has_value());
}

TEST_CASE("FboService::resize rewrites every attachment before rebuilding", "[fbo][seam]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const FboService fboService(storage, backend);

    const auto created = fboService.create(colourAndDepth(320, 240));
    REQUIRE(created.has_value());

    REQUIRE(fboService.resize(created.value(), 1280, 720).has_value());

    const auto* stored = storage.frameBuffers.find(created.value());
    REQUIRE(stored != nullptr);

    for (const auto& attachmentKey : stored->attachments)
    {
        const auto* attachment = storage.bufferAttachments.find(attachmentKey);
        REQUIRE(attachment != nullptr);
        REQUIRE(attachment->width == 1280);
        REQUIRE(attachment->height == 720);
    }

    // Resize is a delete and a rebuild, not an in-place edit: the resize path is the one
    // deleteFbo is reachable from at all, and no gate exercises it.
    REQUIRE(backend.deletedFboIds.size() == 1);
    REQUIRE(backend.fboRequests.size() == 2);
}

TEST_CASE("getAttachmentsOfType filters by type and skips removed attachments", "[fbo][seam]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const FboService fboService(storage, backend);

    const auto created = fboService.create(colourAndDepth(320, 240));
    REQUIRE(created.has_value());

    const auto frameBuffer = *storage.frameBuffers.find(created.value());

    REQUIRE(fboService.getAttachmentsOfType(frameBuffer, FboAttachmentType::Color).size() == 1);
    REQUIRE(fboService.getAttachmentsOfType(frameBuffer, FboAttachmentType::Depth).size() == 1);
    REQUIRE(fboService.getAttachmentsOfType(frameBuffer, FboAttachmentType::Stencil).empty());

    REQUIRE(storage.bufferAttachments.remove(frameBuffer.attachments.front()));

    REQUIRE(fboService.getAttachmentsOfType(frameBuffer, FboAttachmentType::Color).empty());
    REQUIRE(fboService.getAttachmentsOfType(frameBuffer, FboAttachmentType::Depth).size() == 1);
}

TEST_CASE("ShaderService::createShader registers the shader under its name", "[shader][seam]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    ShaderService shaderService(storage, backend);

    const auto descriptor = ShaderDescriptor{.vertexShaderSource = "vs", .fragmentShaderSource = "fs"};
    const auto created = shaderService.createShader("pbr", descriptor);

    REQUIRE(created.has_value());
    REQUIRE(storage.shaders.exists(created.value()));
    REQUIRE(backend.shaderObjectRequests.size() == 1);
    REQUIRE(backend.shaderObjectRequests.front().vertexShaderSource == "vs");

    const auto byName = shaderService.getShaderByName("pbr");
    REQUIRE(byName.has_value());
    REQUIRE(byName.value() == created.value());

    REQUIRE_FALSE(shaderService.getShaderByName("nothing-was-registered-here").has_value());
}

TEST_CASE("a shader the backend refused is not registered under its name", "[shader][seam][expected]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    backend.shaderObjectFailure = "vertex stage: syntax error";
    ShaderService shaderService(storage, backend);

    const auto created = shaderService.createShader("pbr", ShaderDescriptor{});

    REQUIRE_FALSE(created.has_value());

    // The name is in the sentence because the caller asked for a named shader and the backend
    // has no idea what it was called.
    REQUIRE(created.error() == "shader 'pbr' was not created: vertex stage: syntax error");

    // Nothing was stored and nothing answers to the name, so a later getShaderByName cannot hand
    // back a shader with no program behind it.
    REQUIRE(storage.shaders.size() == 0);
    REQUIRE_FALSE(shaderService.getShaderByName("pbr").has_value());
}

TEST_CASE("CubeMapService::create refuses a face that is no longer loaded", "[cubemap][seam][expected]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const CubeMapService cubeMapService(backend, storage);

    const auto front = storage.textures.add(face("front"));
    const auto back = storage.textures.add(face("back"));
    const auto left = storage.textures.add(face("left"));
    const auto right = storage.textures.add(face("right"));
    const auto top = storage.textures.add(face("top"));
    const auto bottom = storage.textures.add(face("bottom"));

    REQUIRE(storage.textures.remove(left));

    const auto created = cubeMapService.create("sky", front, back, left, right, top, bottom);

    REQUIRE_FALSE(created.has_value());
    REQUIRE(created.error() == "cube map 'sky' names a texture that is no longer loaded");
    REQUIRE(backend.cubeMapRequests.empty());
}

TEST_CASE("CubeMapService::create reports what the backend reported and keeps the pixels", "[cubemap][seam][expected]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    backend.cubeMapFailure = "unsupported cube face format";
    const CubeMapService cubeMapService(backend, storage);

    const auto front = storage.textures.add(face("front"));
    const auto back = storage.textures.add(face("back"));
    const auto left = storage.textures.add(face("left"));
    const auto right = storage.textures.add(face("right"));
    const auto top = storage.textures.add(face("top"));
    const auto bottom = storage.textures.add(face("bottom"));

    const auto created = cubeMapService.create("sky", front, back, left, right, top, bottom);

    REQUIRE_FALSE(created.has_value());
    REQUIRE(created.error() == "cube map 'sky' was not created: unsupported cube face format");

    // The GPU never took ownership, so the CPU copies are still the only ones there are.
    REQUIRE_FALSE(storage.textures.find(front)->data.empty());
    REQUIRE(storage.cubeMaps.size() == 0);
}

TEST_CASE("a created cube map takes the faces' pixel data with it", "[cubemap][seam]")
{
    MemoryStorageService storage;
    RecordingRenderBackend backend;
    const CubeMapService cubeMapService(backend, storage);

    const auto front = storage.textures.add(face("front"));
    const auto back = storage.textures.add(face("back"));
    const auto left = storage.textures.add(face("left"));
    const auto right = storage.textures.add(face("right"));
    const auto top = storage.textures.add(face("top"));
    const auto bottom = storage.textures.add(face("bottom"));

    const auto created = cubeMapService.create("sky", front, back, left, right, top, bottom);

    REQUIRE(created.has_value());
    REQUIRE(storage.cubeMaps.exists(created.value()));

    // Face order is part of the contract with the backend, and the service passes six separate
    // handles that are easy to transpose.
    REQUIRE(backend.cubeMapRequests.size() == 1);
    REQUIRE(backend.cubeMapRequests.front().faceNames ==
            std::vector<std::string>{"front", "back", "left", "right", "top", "bottom"});

    const auto* cubeMap = storage.cubeMaps.find(created.value());
    REQUIRE(cubeMap != nullptr);
    REQUIRE(cubeMap->front == front);
    REQUIRE(cubeMap->bottom == bottom);

    // The ownership statement: the GPU has the texels, so the CPU copies go — and they go here
    // rather than inside a backend that would have to const_cast to do it.
    for (const auto& faceKey : {front, back, left, right, top, bottom})
    {
        const auto* texture = storage.textures.find(faceKey);
        REQUIRE(texture != nullptr);
        REQUIRE(texture->data.empty());
    }
}

TEST_CASE("PresenterService records nothing until a presenter is chosen", "[presenter][seam]")
{
    RecordingRenderBackend backend;
    PresenterService presenterService(backend);

    presenterService.record();

    // A level still being built, not a failure: the backend's frame puts the clear colour up.
    REQUIRE(backend.presentRequests.empty());
}

TEST_CASE("PresenterService records the chosen presenter's pass once per call", "[presenter][seam]")
{
    RecordingRenderBackend backend;
    PresenterService presenterService(backend);

    const auto shader = Resource<raceengine::Shader>{.index = 3, .generation = 1};
    const auto output = Resource<raceengine::FboAttachment>{.index = 7, .generation = 2};

    presenterService.setPresenter(Presenter{.output = output, .shader = shader});
    presenterService.record();

    REQUIRE(backend.presentRequests.size() == 1);
    REQUIRE(backend.presentRequests.front().shader == shader);
    REQUIRE(backend.presentRequests.front().attachment == output);

    // Recording is not presenting: the pass reaches the screen when endFrame submits, so two
    // record() calls are two passes and never two presents.
    presenterService.record();
    REQUIRE(backend.presentRequests.size() == 2);
    REQUIRE(backend.endFrameCalls == 0);

    const auto replacementShader = Resource<Shader>{.index = 4, .generation = 1};
    presenterService.setPresenter(Presenter{.output = output, .shader = replacementShader});
    presenterService.record();

    REQUIRE(backend.presentRequests.back().shader == replacementShader);
}
