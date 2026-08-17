#include "ResourceService.h"

#include <fstream>
#include <ozz/base/io/archive.h>
#include <stb_image.h>


ResourceService::ResourceService(
    spdlog::logger& logger,
    MemoryStorageService& memoryStorageService,
    BackgroundWorkerService& backgroundWorkerService,
    GLTFService& gltfService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    backgroundWorkerService(backgroundWorkerService),
    gltfService(gltfService)
{
}

std::expected<std::string, std::string> ResourceService::loadTextFile(const std::string& filePath) const
{
    std::string output;
    std::ifstream fileStream(filePath);

    if (!fileStream.is_open())
    {
        return std::unexpected("Unable to open file with path " + filePath);
    }

    fileStream.seekg(0, std::ios::end);
    output.reserve(fileStream.tellg());
    fileStream.seekg(0, std::ios::beg);

    output.assign((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());

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
        return std::unexpected("Unable to load model with path " + filePath);
    }

    return memoryStorageService.models.add(std::move(model).value());
}

std::expected<Resource<std::unique_ptr<ozz::animation::Skeleton>>, std::string> ResourceService::loadSkeleton(const std::string& filePath) const
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

    return memoryStorageService.skeletons.add(std::move(skeleton));
}


std::expected<Resource<std::unique_ptr<ozz::animation::Animation>>, std::string> ResourceService::loadAnimation(const std::string& filePath) const
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

        return memoryStorageService.textures.add(Texture {
            .name = filePath,
            .format = TextureFormat::RGB,
            .pixelDataType = PixelDataType::Float,
            .width = static_cast<unsigned int>(width),
            .height = static_cast<unsigned int>(height),
            .bitsPerPixel = 96,
            .data = data
        });
    }

    auto* pixels = stbi_load(filePath.c_str(), &width, &height, &channelsInFile, STBI_rgb_alpha);

    if (!pixels)
    {
        return std::unexpected("Unable to load image with path " + filePath + ": " + stbi_failure_reason());
    }

    const auto byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    auto data = std::vector<unsigned char>(pixels, pixels + byteCount);

    stbi_image_free(pixels);

    return memoryStorageService.textures.add(Texture {
        .name = filePath,
        .format = TextureFormat::RGBA,
        .pixelDataType = PixelDataType::UnsignedByte,
        .width = static_cast<unsigned int>(width),
        .height = static_cast<unsigned int>(height),
        .bitsPerPixel = 32,
        .data = data
    });
}


AsyncResult<std::string> ResourceService::loadTextFileAsync(std::string filePath) const
{
    logger.info("Loading file: {}", filePath);

    return backgroundWorkerService.submit(
        [this, filePath = std::move(filePath)] { return loadTextFile(filePath); });
}

AsyncResult<Resource<Model>> ResourceService::loadModelAsync(std::string filePath) const
{
    logger.info("Loading model: {}", filePath);

    return backgroundWorkerService.submit(
        [this, filePath = std::move(filePath)] { return loadModel(filePath); });
}

AsyncResult<Resource<std::unique_ptr<ozz::animation::Skeleton>>> ResourceService::loadSkeletonAsync(std::string filePath) const
{
    logger.info("Loading skeleton: {}", filePath);

    return backgroundWorkerService.submit(
        [this, filePath = std::move(filePath)] { return loadSkeleton(filePath); });
}

AsyncResult<Resource<std::unique_ptr<ozz::animation::Animation>>> ResourceService::loadAnimationAsync(std::string filePath) const
{
    logger.info("Loading skeleton: {}", filePath);

    return backgroundWorkerService.submit(
        [this, filePath = std::move(filePath)] { return loadAnimation(filePath); });
}

AsyncResult<Resource<Texture>> ResourceService::loadTextureAsync(std::string filePath) const
{
    logger.info("Loading image: {}", filePath);

    return backgroundWorkerService.submit(
        [this, filePath = std::move(filePath)] { return loadTexture(filePath); });
}

