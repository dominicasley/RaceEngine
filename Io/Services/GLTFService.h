#pragma once

#include <spdlog/logger.h>
#include <optional>
#include <vector>
#include <tiny_gltf.h>
#include "Shared/Services/MemoryStorageService.h"
#include "Graphics/Models/Scene/Model.h"
#include "Graphics/Models/Scene/TextureFormat.h"
#include "Graphics/Models/Scene/Material.h"
#include "Graphics/Models/Scene/Texture.h"

class GLTFService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;

public:
    explicit GLTFService(spdlog::logger& logger, MemoryStorageService& memoryStorageService);
    [[nodiscard]] Model gltfModelToInternal(const std::string& fileName, const tinygltf::Model& tinyGltfModel) const;
    [[nodiscard]] std::optional<Model> loadModelFromFile(const std::string& filePath) const;
    [[nodiscard]] std::optional<PrimitiveAttributeType> toAttributeType(const std::string& attributeName) const;
    [[nodiscard]] TextureFormat toTextureFormat(int format) const;
    [[nodiscard]] Texture getImageFromIndex(const tinygltf::Model& model, int index) const;
    [[nodiscard]] std::optional<VertexIndicesType> toVertexIndicesType(int componentType) const;
    void processNode(Model& model, const tinygltf::Model& tinyGltfModel, const tinygltf::Node &node, const glm::mat4 parentTransform) const;
};


