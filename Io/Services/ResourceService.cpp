#include "ResourceService.h"

#include <utility>
#include <fstream>


ResourceService::ResourceService(
    spdlog::logger& logger,
    MemoryStorageService& memoryStorageService,
    BackgroundWorkerService& backgroundWorkerService) :
    logger(logger),
    memoryStorageService(memoryStorageService),
    backgroundWorkerService(backgroundWorkerService)
{
}

tinygltf::Model* ResourceService::loadModel(std::string filePath)
{
    if (memoryStorageService.modelExists(filePath))
    {
        logger.info("Model {} already loaded in memory", filePath);
        return memoryStorageService.getModel(filePath);
    }

    tinygltf::TinyGLTF gltfLoader;

    bool result;
    std::string error;
    std::string warning;
    tinygltf::Model model;

    auto fileExtension = filePath.substr(filePath.find_last_of('.') + 1);
    if (fileExtension == "gltf")
    {
        result = gltfLoader.LoadASCIIFromFile(&model, &error, &warning, filePath);
    }
    else if (fileExtension == "glb")
    {
        result = gltfLoader.LoadBinaryFromFile(&model, &error, &warning, filePath);
    }
    else
    {
        logger.error("Unknown extension {} when loading model with path {}", fileExtension, filePath);
        return nullptr;
    }

    if (result)
    {
        logger.info("Loaded model: {}", filePath);
    }
    else
    {
        logger.error("Failed to load model: {}", filePath);
        return nullptr;
    }

    if (!warning.empty())
    {
        logger.warn(warning);
    }

    if (!error.empty())
    {
        logger.error(error);
    }

    return memoryStorageService.addModel(filePath, model);
}

std::string ResourceService::loadTextFile(std::string filePath)
{
    std::string output;
    std::ifstream fileStream(filePath);

    fileStream.seekg(0, std::ios::end);
    output.reserve(fileStream.tellg());
    fileStream.seekg(0, std::ios::beg);

    output.assign((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());

    return output;
}

const Observable<tinygltf::Model*>& ResourceService::loadModelAsync(std::string filePath)
{
    logger.info("Loading model: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadModel, this, filePath))
        ->asObservable();
}

const Observable<std::string>& ResourceService::loadTextFileAsync(std::string filePath)
{
    logger.info("Loading file: {}", filePath);

    return BackgroundWorkerService::registerTask(
        std::async(std::launch::async, &ResourceService::loadTextFile, this, filePath))
        ->asObservable();
}
