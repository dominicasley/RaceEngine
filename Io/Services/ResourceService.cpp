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

std::string ResourceService::loadTextFile(const std::string& filePath) const
{
    std::string output;
    std::ifstream fileStream(filePath);

    fileStream.seekg(0, std::ios::end);
    output.reserve(fileStream.tellg());
    fileStream.seekg(0, std::ios::beg);

    output.assign((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());

    return output;
}

Resource<Model> ResourceService::loadModel(const std::string& filePath) const
{
    std::optional<Model> model;
    auto fileExtension = filePath.substr(filePath.find_last_of('.') + 1);
    if (fileExtension == "gltf" || fileExtension == "glb")
    {
        model = gltfService.loadModelFromFile(filePath);
    }

    return memoryStorageService.models.add(model.value());
}

Resource<std::unique_ptr<ozz::animation::Skeleton>> ResourceService::loadSkeleton(const std::string& filePath) const
{
    ozz::io::File file(filePath.c_str(), "rb");

    if (!file.opened())
    {
        logger.error("Failed to open skeleton file {}", filePath);
    }

    ozz::io::IArchive archive(&file);
    if (!archive.TestTag<ozz::animation::Skeleton>())
    {
        logger.error("Failed to load skeleton instance from file {}", filePath);
    }

    auto skeleton = std::make_unique<ozz::animation::Skeleton>();
    archive >> *skeleton;

    return memoryStorageService.skeletons.takeOwnership(skeleton);
}


Resource<std::unique_ptr<ozz::animation::Animation>> ResourceService::loadAnimation(const std::string& filePath) const
{

    ozz::io::File file(filePath.c_str(), "rb");

    if (!file.opened())
    {
        logger.error("Failed to open skeleton file {}", filePath);
    }

    ozz::io::IArchive archive(&file);
    if (!archive.TestTag<ozz::animation::Animation>())
    {
        logger.error("Failed to load skeleton instance from file {}", filePath);
    }

    auto animation = std::make_unique<ozz::animation::Animation>();
    archive >> *animation;

    return memoryStorageService.animations.takeOwnership(animation);
}

Resource<Texture> ResourceService::loadTexture(const std::string& filePath) const
{
    int width = 0;
    int height = 0;
    int channelsInFile = 0;

    if (filePath.ends_with(".hdr"))
    {
        auto* pixels = stbi_loadf(filePath.c_str(), &width, &height, &channelsInFile, 3);

        if (!pixels)
        {
            logger.error("Unable to load image with path {}: {}", filePath, stbi_failure_reason());
            return memoryStorageService.textures.add(Texture {
                .name = filePath,
                .format = TextureFormat::Unknown,
                .pixelDataType = PixelDataType::Float,
                .width = 0,
                .height = 0,
                .bitsPerPixel = 0,
                .data = {}
            });
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
        logger.error("Unable to load image with path {}: {}", filePath, stbi_failure_reason());
        return memoryStorageService.textures.add(Texture {
            .name = filePath,
            .format = TextureFormat::Unknown,
            .pixelDataType = PixelDataType::UnsignedByte,
            .width = 0,
            .height = 0,
            .bitsPerPixel = 0,
            .data = {}
        });
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


const Observable<std::string>& ResourceService::loadTextFileAsync(std::string filePath) const
{
    logger.info("Loading file: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadTextFile, this, filePath))
        ->asObservable();
}

const Observable<Resource<Model>>& ResourceService::loadModelAsync(std::string filePath) const
{
    logger.info("Loading model: {}", filePath);

    return BackgroundWorkerService::registerTask(
            std::async(std::launch::async, &ResourceService::loadModel, this, filePath))
        ->asObservable();
}

const Observable<Resource<std::unique_ptr<ozz::animation::Skeleton>>>& ResourceService::loadSkeletonAsync(std::string filePath) const
{
    logger.info("Loading skeleton: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadSkeleton, this, filePath))
        ->asObservable();
}

const Observable<Resource<std::unique_ptr<ozz::animation::Animation>>>& ResourceService::loadAnimationAsync(std::string filePath) const
{
    logger.info("Loading skeleton: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadAnimation, this, filePath))
        ->asObservable();
}

const Observable<Resource<Texture>>& ResourceService::loadTextureAsync(std::string filePath) const
{
    logger.info("Loading image: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadTexture, this, filePath))
        ->asObservable();
}

