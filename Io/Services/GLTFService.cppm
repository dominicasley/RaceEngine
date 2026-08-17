module;

#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/logger.h>
#include <tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.io:GLTFService;

import :AccessorUtility;
import raceengine.graphics.models;
import raceengine.shared;

// Defined in Io/ThirdPartyImpl.cpp; see the note there for why the loader call
// cannot live in this unit.
extern "C++" bool raceengineLoadTinyGltf(
    tinygltf::Model& model,
    std::string& error,
    std::string& warning,
    const std::string& filePath,
    bool binary);

namespace raceengine
{

export class GLTFService
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

GLTFService::GLTFService(spdlog::logger& logger, MemoryStorageService& memoryStorageService) :
    logger(logger),
    memoryStorageService(memoryStorageService)
{

}

std::optional<Model> GLTFService::loadModelFromFile(const std::string& filePath) const
{
    bool result;
    std::string error;
    std::string warning;
    tinygltf::Model model;

    auto fileExtension = filePath.substr(filePath.find_last_of('.') + 1);
    if (fileExtension == "gltf")
    {
        result = raceengineLoadTinyGltf(model, error, warning, filePath, false);
    }
    else if (fileExtension == "glb")
    {
        result = raceengineLoadTinyGltf(model, error, warning, filePath, true);
    }
    else
    {
        logger.error("Unknown extension {} when loading model with path {}", fileExtension, filePath);
        return std::nullopt;
    }

    if (result)
    {
        logger.info("Loaded model: {}", filePath);
    }
    else
    {
        logger.error("Failed to load model: {}", filePath);
        return std::nullopt;
    }

    if (!warning.empty())
    {
        logger.warn(warning);
    }

    if (!error.empty())
    {
        logger.error(error);
    }

    return gltfModelToInternal(filePath, model);
}

void GLTFService::processNode(Model& model, const tinygltf::Model& tinyGltfModel, const tinygltf::Node& node, const glm::mat4 parentTransform) const
{
    auto transform = parentTransform * ((node.translation.size() == 3 ?
                                         glm::translate(
                                             glm::mat4(1.0f),
                                             glm::vec3(node.translation[0], node.translation[1], node.translation[2])) : glm::mat4(1.0f)) *
                                        (node.rotation.size() == 4 ?
                                         glm::mat4_cast(
                                             glm::quat(static_cast<float>(node.rotation[3]),
                                                       static_cast<float>(node.rotation[0]),
                                                       static_cast<float>(node.rotation[1]),
                                                       static_cast<float>(node.rotation[2]))) : glm::mat4(1.0f)) *
                                        (node.scale.size() == 3 ?
                                         glm::scale(
                                             glm::mat4(1.0f),
                                             glm::vec3(node.scale[0], node.scale[1], node.scale[2])) : glm::mat4(1.0f)));

    if (node.mesh == -1)
    {
        for (const auto& child: node.children)
        {
            processNode(model, tinyGltfModel, tinyGltfModel.nodes[static_cast<size_t>(child)], transform);
        }

        return;
    }

    const auto tinyGltfMesh = tinyGltfModel.meshes[static_cast<size_t>(node.mesh)];

    Mesh mesh;
    mesh.name = tinyGltfMesh.name;
    mesh.modelMatrix = transform;

    if (node.skin != -1)
    {
        auto skin = tinyGltfModel.skins[static_cast<size_t>(node.skin)];
        for (size_t j = 0; j < skin.joints.size(); j++)
        {
            const auto jointNodeIndex = static_cast<size_t>(skin.joints[j]);
            if (tinyGltfModel.nodes[jointNodeIndex].name.empty())
            {
                mesh.skin["node_" + std::to_string(skin.joints[j])] = static_cast<int>(j);
            }
            else
            {
                mesh.skin[tinyGltfModel.nodes[jointNodeIndex].name] = static_cast<int>(j);
            }
        }

        mesh.inverseBindPoseTransforms = AccessorUtility::get<std::vector<glm::mat4>>(
            tinyGltfModel,
            tinyGltfModel.accessors[static_cast<size_t>(skin.inverseBindMatrices)]);
    }

    for (const auto& primitive: tinyGltfMesh.primitives)
    {
        if (primitive.indices < 0)
        {
            logger.warn("Skipping non-indexed primitive in mesh {}", tinyGltfMesh.name);
            continue;
        }

        auto indexAccessor = tinyGltfModel.accessors[static_cast<size_t>(primitive.indices)];

        auto meshPrimitive = MeshPrimitive{
            .mode = primitive.mode == -1 ? TINYGLTF_MODE_TRIANGLES : primitive.mode,
            .material = primitive.material != -1 ?
                        std::optional<Resource<Material>>(model.materials[static_cast<size_t>(primitive.material)]) : std::nullopt,
            .elementCount = indexAccessor.count,
            .byteOffset = indexAccessor.byteOffset,
            .componentType = indexAccessor.componentType,
            .meshBufferIndex = indexAccessor.bufferView
        };

        for (auto& attribute: primitive.attributes)
        {
            auto attributeType = toAttributeType(attribute.first);

            if (!attributeType.has_value())
            {
                continue;
            }

            auto accessor = tinyGltfModel.accessors[static_cast<size_t>(attribute.second)];
            auto byteStride = accessor.ByteStride(tinyGltfModel.bufferViews[static_cast<size_t>(accessor.bufferView)]);

            meshPrimitive.attributes.push_back(MeshPrimitiveAttribute{
                .size = accessor.type != TINYGLTF_TYPE_SCALAR ? accessor.type : 1,
                .type = accessor.type,
                .componentType = accessor.componentType,
                .stride = byteStride,
                .bufferIndex = accessor.bufferView,
                .attributeType = attributeType,
                .normalized = accessor.normalized,
                .offset = accessor.byteOffset,
            });
        }

        mesh.meshPrimitives.push_back(meshPrimitive);
    }

    model.meshes.push_back(memoryStorageService.meshes.add(mesh));

    for (const auto& child: node.children)
    {
        processNode(model, tinyGltfModel, tinyGltfModel.nodes[static_cast<size_t>(child)], transform);
    }
}

Model GLTFService::gltfModelToInternal(const std::string& filePath, const tinygltf::Model& tinyGltfModel) const
{
    logger.info("Processing model: {}", filePath);
    Model model;

    std::map<int, Resource<Texture>> textureMap;

    for (const auto& texture: tinyGltfModel.textures)
    {
        if (texture.source < 0)
        {
            continue;
        }

        auto image = getImageFromIndex(tinyGltfModel, texture.source);
        textureMap.insert_or_assign(texture.source, memoryStorageService.textures.add(image));
    }

    for (const auto& tinyGltfMaterial: tinyGltfModel.materials)
    {
        std::optional<Resource<Texture>> albedoTexturePtr;
        std::optional<Resource<Texture>> metallicRoughnessTexturePtr;
        std::optional<Resource<Texture>> normalTexturePtr;
        std::optional<Resource<Texture>> occlusionTexturePtr;
        std::optional<Resource<Texture>> emissiveTexturePtr;

        if (tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index != -1)
        {
            albedoTexturePtr = textureMap[tinyGltfModel.textures[static_cast<size_t>(tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index)].source];
        }

        if (tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index != -1)
        {
            metallicRoughnessTexturePtr = textureMap[tinyGltfModel.textures[static_cast<size_t>(tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index)].source];
        }

        if (tinyGltfMaterial.normalTexture.index != -1)
        {
            normalTexturePtr = textureMap[tinyGltfModel.textures[static_cast<size_t>(tinyGltfMaterial.normalTexture.index)].source];
        }

        if (tinyGltfMaterial.occlusionTexture.index != -1)
        {
            occlusionTexturePtr = textureMap[tinyGltfModel.textures[static_cast<size_t>(tinyGltfMaterial.occlusionTexture.index)].source];
        }

        if (tinyGltfMaterial.emissiveTexture.index != -1)
        {
            emissiveTexturePtr = textureMap[tinyGltfModel.textures[static_cast<size_t>(tinyGltfMaterial.emissiveTexture.index)].source];
        }

        auto material = memoryStorageService.materials.add(Material{
            .baseColour = glm::vec4(
                tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[0],
                tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[1],
                tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[2],
                tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[3]
                ),
            .metalness = static_cast<float>(tinyGltfMaterial.pbrMetallicRoughness.metallicFactor),
            .roughness = static_cast<float>(tinyGltfMaterial.pbrMetallicRoughness.roughnessFactor),
            .opaque = true,
            .albedo = albedoTexturePtr,
            .metallicRoughness = metallicRoughnessTexturePtr,
            .normal = normalTexturePtr,
            .occlusion = occlusionTexturePtr,
            .emissive = emissiveTexturePtr,
        });

        model.materials.push_back(material);
    }

    // bufferView.target is optional in glTF; infer it from usage when absent:
    // views referenced by primitive indices are element array buffers, everything else uploads as a vertex buffer.
    std::set<int> indexBufferViews;
    for (const auto& tinyGltfMesh: tinyGltfModel.meshes)
    {
        for (const auto& primitive: tinyGltfMesh.primitives)
        {
            if (primitive.indices < 0)
            {
                continue;
            }

            const auto bufferView = tinyGltfModel.accessors[static_cast<size_t>(primitive.indices)].bufferView;
            if (bufferView >= 0)
            {
                indexBufferViews.insert(bufferView);
            }
        }
    }

    for (auto i = 0; std::cmp_less(i, tinyGltfModel.bufferViews.size()); i++)
    {
        const auto& bufferView = tinyGltfModel.bufferViews[static_cast<size_t>(i)];

        auto target = bufferView.target;
        if (target == 0)
        {
            target = indexBufferViews.contains(i)
                         ? TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER
                         : TINYGLTF_TARGET_ARRAY_BUFFER;
        }

        // Store only this view's slice of the binary blob; accessor byte offsets are
        // relative to the bufferView, so they keep working against the sliced data.
        const auto& blob = tinyGltfModel.buffers[static_cast<size_t>(bufferView.buffer)].data;
        const auto sliceBegin = blob.begin() + static_cast<std::ptrdiff_t>(bufferView.byteOffset);

        model.meshBuffers.push_back(MeshBuffer{
            .target = target,
            .length = bufferView.byteLength,
            .offset = 0,
            .stride = bufferView.byteStride,
            .data = std::vector<unsigned char>(sliceBegin, sliceBegin + static_cast<std::ptrdiff_t>(bufferView.byteLength))
        });
    }

    const auto sceneIndex = tinyGltfModel.defaultScene < 0 ? 0 : tinyGltfModel.defaultScene;
    for (auto& sceneNode: tinyGltfModel.scenes[static_cast<size_t>(sceneIndex)].nodes)
    {
        processNode(model, tinyGltfModel, tinyGltfModel.nodes[static_cast<size_t>(sceneNode)], glm::mat4(1.0));
    }

    logger.info("Processed model: {}", filePath);
    return model;
}

std::optional<PrimitiveAttributeType> GLTFService::toAttributeType(const std::string& attributeName) const
{
    if (attributeName == "POSITION")
    {
        return PrimitiveAttributeType::Position;
    }
    else if (attributeName.starts_with("TEXCOORD"))
    {
        return PrimitiveAttributeType::TextureCoordinate;
    }
    else if (attributeName == "NORMAL")
    {
        return PrimitiveAttributeType::Normal;
    }
    else if (attributeName == "TANGENT")
    {
        return PrimitiveAttributeType::Tangent;
    }
    else if (attributeName.starts_with("JOINTS"))
    {
        return PrimitiveAttributeType::Joint;
    }
    else if (attributeName.starts_with("WEIGHTS"))
    {
        return PrimitiveAttributeType::SkinWeight;
    }

    logger.warn("Unhandled attribute type: {}", attributeName);

    return {};
}

TextureFormat GLTFService::toTextureFormat(int format) const
{
    switch (format)
    {
        case 1:
            return TextureFormat::R;
        case 2:
            return TextureFormat::RG;
        case 3:
            return TextureFormat::RGB;
        case 4:
        default:
            return TextureFormat::RGBA;
    }
}

Texture GLTFService::getImageFromIndex(const tinygltf::Model& model, int index) const
{
    auto image = model.images[static_cast<size_t>(index)];

    auto width = static_cast<unsigned int>(image.width);
    auto height = static_cast<unsigned int>(image.height);

    return Texture {
        .name = image.name,
        .format = toTextureFormat(image.component),
        .pixelDataType = image.bits == 16 ? PixelDataType::UnsignedShort : PixelDataType::UnsignedByte,
        .width = width,
        .height = height,
        .data = image.image
    };
}

std::optional<VertexIndicesType> GLTFService::toVertexIndicesType(int componentType) const
{
    switch (componentType)
    {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return VertexIndicesType::UnsignedByte;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return VertexIndicesType::UnsignedShort;
        case TINYGLTF_COMPONENT_TYPE_INT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return VertexIndicesType::UnsignedInt;
        default:
            return {};
    }
}

} // namespace raceengine
