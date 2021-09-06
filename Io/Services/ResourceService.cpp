#include "ResourceService.h"

#include <fstream>
#include <ozz/base/io/archive.h>
#include <FreeImage.h>


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
    auto format = FreeImage_GetFileType(filePath.c_str(), 0);

    if (format == FIF_UNKNOWN)
    {
        format = FreeImage_GetFIFFromFilename(filePath.c_str());
    }

    if (!FreeImage_FIFSupportsReading(format))
    {
        logger.error("No read support for image format {}: {}", format, filePath);
    }

    auto image = FreeImage_Load(format, filePath.c_str());

    if (!image)
    {
        logger.error("Unable to load image with path {}", filePath);
    }

    auto imageData = FreeImage_GetBits(image);
    auto size = FreeImage_GetMemorySize(image);
    auto bitsPerPixel = FreeImage_GetBPP(image);
    auto imageWidth = FreeImage_GetWidth(image);
    auto imageHeight = FreeImage_GetHeight(image);
    auto data = std::vector(imageData, imageData + size);

    FreeImage_Unload(image);

    return memoryStorageService.textures.add(Texture {
        .name = filePath,
        .format = bitsPerPixel == 24 ? TextureFormat::BGR :
                  bitsPerPixel == 32 ? TextureFormat::BGRA :
                  bitsPerPixel == 96 ? TextureFormat::RGB :
                  bitsPerPixel == 128 ? TextureFormat::RGBA :
                  TextureFormat::Unknown,
        .pixelDataType = bitsPerPixel == 96 ? PixelDataType::Float : PixelDataType::UnsignedByte,
        .width = imageWidth,
        .height = imageHeight,
        .bitsPerPixel = bitsPerPixel,
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

