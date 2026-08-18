module;

#include <expected>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/io/archive.h>
#include <spdlog/logger.h>
#include <stb_image.h>

export module raceengine.io:ResourceService;

import :GLTFService;
import raceengine.async;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class ResourceService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;
    BackgroundWorkerService& backgroundWorkerService;
    GLTFService& gltfService;

public:
    explicit ResourceService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                             BackgroundWorkerService& backgroundWorkerService, GLTFService& gltfService);
    [[nodiscard]] std::expected<std::string, std::string> loadTextFile(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<Model>, std::string> loadModel(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<std::unique_ptr<ozz::animation::Skeleton>>, std::string>
    loadSkeleton(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<std::unique_ptr<ozz::animation::Animation>>, std::string>
    loadAnimation(const std::string& filePath) const;
    [[nodiscard]] std::expected<Resource<Texture>, std::string> loadTexture(const std::string& filePath) const;
    [[nodiscard]] AsyncResult<std::string> loadTextFileAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<Model>> loadModelAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<Texture>> loadTextureAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<std::unique_ptr<ozz::animation::Skeleton>>>
    loadSkeletonAsync(std::string filePath) const;
    [[nodiscard]] AsyncResult<Resource<std::unique_ptr<ozz::animation::Animation>>>
    loadAnimationAsync(std::string filePath) const;
};

ResourceService::ResourceService(spdlog::logger& logger, MemoryStorageService& memoryStorageService,
                                 BackgroundWorkerService& backgroundWorkerService, GLTFService& gltfService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    backgroundWorkerService(backgroundWorkerService),
    gltfService(gltfService)
{
}

// Opened binary and read by size, because the size is the only thing that can say the read
// finished. The previous shape streamed through an istreambuf_iterator, which works on the
// streambuf directly and never touches the stream's state bits: a read that stopped early — a
// truncated file, a disk giving up mid-shader — produced a short string and reported success, and
// what the caller then compiled was half a shader.
std::expected<std::string, std::string> ResourceService::loadTextFile(const std::string& filePath) const
{
    std::ifstream fileStream(filePath, std::ios::binary | std::ios::ate);

    if (!fileStream.is_open())
    {
        return std::unexpected("Unable to open file with path " + filePath);
    }

    const auto declaredSize = fileStream.tellg();
    if (declaredSize < 0)
    {
        return std::unexpected("Unable to determine the size of file with path " + filePath);
    }

    auto output = std::string(static_cast<size_t>(declaredSize), '\0');
    fileStream.seekg(0, std::ios::beg);
    fileStream.read(output.data(), declaredSize);

    if (fileStream.gcount() != declaredSize)
    {
        return std::unexpected("Read " + std::to_string(fileStream.gcount()) + " of the " +
                               std::to_string(declaredSize) + " bytes of file with path " + filePath);
    }

    return output;
}

std::expected<Resource<Model>, std::string> ResourceService::loadModel(const std::string& filePath) const
{
    auto fileExtension = filePath.substr(filePath.find_last_of('.') + 1);
    if (fileExtension != "gltf" && fileExtension != "glb")
    {
        return std::unexpected("Unknown extension " + fileExtension + " when loading model with path " + filePath);
    }

    auto model = gltfService.loadModelFromFile(filePath);
    if (!model)
    {
        // The importer's own sentence, not a replacement for it: it names the accessor, the
        // buffer view or the index that is wrong, and "Unable to load model with path X" was
        // this layer throwing that away and restating what the caller already knew.
        return std::unexpected("Unable to load model with path " + filePath + ": " + model.error());
    }

    return memoryStorageService.models.add(std::move(model).value());
}

std::expected<Resource<std::unique_ptr<ozz::animation::Skeleton>>, std::string>
ResourceService::loadSkeleton(const std::string& filePath) const
{
    ozz::io::File file(filePath.c_str(), "rb");

    if (!file.opened())
    {
        return std::unexpected("Failed to open skeleton file " + filePath);
    }

    ozz::io::IArchive archive(&file);
    if (!archive.TestTag<ozz::animation::Skeleton>())
    {
        return std::unexpected("Failed to load skeleton instance from file " + filePath);
    }

    auto skeleton = std::make_unique<ozz::animation::Skeleton>();
    archive >> *skeleton;

    // ozz states plainly that its archives detect nothing while reading and that stream
    // integrity — truncation, corruption — is the caller's to check. The tag above proves the
    // file starts as a skeleton; this is what proves one came out. Without it a truncated
    // archive yielded a joint-less skeleton that every later stage treated as a mesh with
    // nothing to animate.
    if (skeleton->num_joints() <= 0)
    {
        return std::unexpected("Skeleton read from " + filePath +
                               " carries no joints; the archive is truncated or "
                               "does not hold what its tag claims");
    }

    return memoryStorageService.skeletons.add(std::move(skeleton));
}

std::expected<Resource<std::unique_ptr<ozz::animation::Animation>>, std::string>
ResourceService::loadAnimation(const std::string& filePath) const
{

    ozz::io::File file(filePath.c_str(), "rb");

    if (!file.opened())
    {
        return std::unexpected("Failed to open animation file " + filePath);
    }

    ozz::io::IArchive archive(&file);
    if (!archive.TestTag<ozz::animation::Animation>())
    {
        return std::unexpected("Failed to load animation instance from file " + filePath);
    }

    auto animation = std::make_unique<ozz::animation::Animation>();
    archive >> *animation;

    // Same reading as the skeleton above: the archive reports nothing, so the object it produced
    // is the evidence. A clip with no tracks or no duration cannot be sampled, and ozz's
    // SamplingJob would refuse it once per frame from inside the draw path instead.
    if (animation->num_tracks() <= 0 || animation->duration() <= 0.0f)
    {
        return std::unexpected("Animation read from " + filePath + " carries " +
                               std::to_string(animation->num_tracks()) + " track(s) over " +
                               std::to_string(animation->duration()) +
                               "s; the archive is truncated or does not hold what its tag claims");
    }

    return memoryStorageService.animations.add(std::move(animation));
}

std::expected<Resource<Texture>, std::string> ResourceService::loadTexture(const std::string& filePath) const
{
    int width = 0;
    int height = 0;
    int channelsInFile = 0;

    if (filePath.ends_with(".hdr"))
    {
        auto* pixels = stbi_loadf(filePath.c_str(), &width, &height, &channelsInFile, 3);

        if (!pixels)
        {
            return std::unexpected("Unable to load image with path " + filePath + ": " + stbi_failure_reason());
        }

        const auto byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 * sizeof(float);
        const auto* bytes = reinterpret_cast<const unsigned char*>(pixels);
        auto data = std::vector<unsigned char>(bytes, bytes + byteCount);

        stbi_image_free(pixels);

        return memoryStorageService.textures.add(Texture{.name = filePath,
                                                         .format = TextureFormat::RGB,
                                                         .pixelDataType = PixelDataType::Float,
                                                         .width = static_cast<unsigned int>(width),
                                                         .height = static_cast<unsigned int>(height),
                                                         .bitsPerPixel = 96,
                                                         .data = data});
    }

    auto* pixels = stbi_load(filePath.c_str(), &width, &height, &channelsInFile, STBI_rgb_alpha);

    if (!pixels)
    {
        return std::unexpected("Unable to load image with path " + filePath + ": " + stbi_failure_reason());
    }

    const auto byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    auto data = std::vector<unsigned char>(pixels, pixels + byteCount);

    stbi_image_free(pixels);

    return memoryStorageService.textures.add(Texture{.name = filePath,
                                                     .format = TextureFormat::RGBA,
                                                     .pixelDataType = PixelDataType::UnsignedByte,
                                                     .width = static_cast<unsigned int>(width),
                                                     .height = static_cast<unsigned int>(height),
                                                     .bitsPerPixel = 32,
                                                     .data = data});
}

AsyncResult<std::string> ResourceService::loadTextFileAsync(std::string filePath) const
{
    logger.info("Loading file: {}", filePath);

    return backgroundWorkerService.submit([this, filePath = std::move(filePath)] { return loadTextFile(filePath); });
}

AsyncResult<Resource<Model>> ResourceService::loadModelAsync(std::string filePath) const
{
    logger.info("Loading model: {}", filePath);

    return backgroundWorkerService.submit([this, filePath = std::move(filePath)] { return loadModel(filePath); });
}

AsyncResult<Resource<std::unique_ptr<ozz::animation::Skeleton>>>
ResourceService::loadSkeletonAsync(std::string filePath) const
{
    logger.info("Loading skeleton: {}", filePath);

    return backgroundWorkerService.submit([this, filePath = std::move(filePath)] { return loadSkeleton(filePath); });
}

AsyncResult<Resource<std::unique_ptr<ozz::animation::Animation>>>
ResourceService::loadAnimationAsync(std::string filePath) const
{
    logger.info("Loading animation: {}", filePath);

    return backgroundWorkerService.submit([this, filePath = std::move(filePath)] { return loadAnimation(filePath); });
}

AsyncResult<Resource<Texture>> ResourceService::loadTextureAsync(std::string filePath) const
{
    logger.info("Loading image: {}", filePath);

    return backgroundWorkerService.submit([this, filePath = std::move(filePath)] { return loadTexture(filePath); });
}

} // namespace raceengine
