#include "ResourceService.h"

#include <fstream>
#include <ozz/base/io/archive.h>


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

std::string ResourceService::loadTextFile(const std::string& filePath)
{
    std::string output;
    std::ifstream fileStream(filePath);

    fileStream.seekg(0, std::ios::end);
    output.reserve(fileStream.tellg());
    fileStream.seekg(0, std::ios::beg);

    output.assign((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());

    return output;
}

Model* ResourceService::loadModel(std::string filePath)
{
    if (memoryStorageService.models.exists(filePath))
    {
        logger.info("Model {} already loaded in memory", filePath);
        return memoryStorageService.models.get(filePath);
    }

    std::optional<Model> model;
    auto fileExtension = filePath.substr(filePath.find_last_of('.') + 1);
    if (fileExtension == "gltf" || fileExtension == "glb")
    {
        model = gltfService.loadModelFromFile(filePath);
    }
    else
    {
        return nullptr;
    }

    if (!model.has_value())
    {
        return nullptr;
    }

    return memoryStorageService.models.add(filePath, model.value());
}

ozz::animation::Skeleton* ResourceService::loadSkeleton(const std::string& filePath)
{
    if (memoryStorageService.skeletons.exists(filePath))
    {
        logger.info("Skeleton {} already loaded in memory", filePath);
        return memoryStorageService.skeletons.get(filePath);
    }

    ozz::io::File file(filePath.c_str(), "rb");

    if (!file.opened()) {
        logger.error("Failed to open skeleton file {}", filePath);
        return nullptr;
    }

    ozz::io::IArchive archive(&file);
    if (!archive.TestTag<ozz::animation::Skeleton>()) {
        logger.error("Failed to load skeleton instance from file {}", filePath);
        return nullptr;
    }

    auto skeleton = std::make_unique<ozz::animation::Skeleton>();
    archive >> *(skeleton.get());

    return memoryStorageService.skeletons.takeOwnership(filePath, skeleton);
}


ozz::animation::Animation* ResourceService::loadAnimation(const std::string& filePath)
{
    if (memoryStorageService.animations.exists(filePath))
    {
        logger.info("Animation {} already loaded in memory", filePath);
        return memoryStorageService.animations.get(filePath);
    }

    ozz::io::File file(filePath.c_str(), "rb");

    if (!file.opened()) {
        logger.error("Failed to open skeleton file {}", filePath);
        return nullptr;
    }

    ozz::io::IArchive archive(&file);
    if (!archive.TestTag<ozz::animation::Animation>()) {
        logger.error("Failed to load skeleton instance from file {}", filePath);
        return nullptr;
    }

    auto animation = std::make_unique<ozz::animation::Animation>();
    archive >> *(animation.get());

    return memoryStorageService.animations.takeOwnership(filePath, animation);
}

const Observable<std::string>& ResourceService::loadTextFileAsync(std::string filePath)
{
    logger.info("Loading file: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadTextFile, this, filePath))
        ->asObservable();
}

const Observable<Model*>& ResourceService::loadModelAsync(std::string filePath)
{
    logger.info("Loading model: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadModel, this, filePath))
        ->asObservable();
}

const Observable<ozz::animation::Skeleton*>& ResourceService::loadSkeletonAsync(std::string filePath)
{
    logger.info("Loading skeleton: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadSkeleton, this, filePath))
        ->asObservable();
}

const Observable<ozz::animation::Animation*>& ResourceService::loadAnimationAsync(std::string filePath)
{
    logger.info("Loading skeleton: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadAnimation, this, filePath))
        ->asObservable();
}
