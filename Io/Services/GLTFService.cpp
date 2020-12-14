#include "GLTFService.h"
#include "../Utility/AccessorUtility.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

GLTFService::GLTFService(spdlog::logger& logger, MemoryStorageService& memoryStorageService) :
    logger(logger),
    memoryStorageService(memoryStorageService)
{

}

std::optional<Model> GLTFService::loadModelFromFile(const std::string& filePath) const
{
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

Model GLTFService::gltfModelToInternal(std::string filePath, const tinygltf::Model& tinyGltfModel) const
{
    Model model;

    for (const auto& node : tinyGltfModel.nodes)
    {
        if (node.mesh == -1)
        {
            continue;
        }

        const auto tinyGltfMesh = tinyGltfModel.meshes[node.mesh];

        Mesh mesh;
        mesh.name = tinyGltfMesh.name;

        if (node.skin != -1)
        {
            auto skin = tinyGltfModel.skins[node.skin];
            for (auto j = 0; j < tinyGltfModel.skins[node.skin].joints.size(); j++)
            {
                if (tinyGltfModel.nodes[skin.joints[j]].name.empty())
                {
                    mesh.skin["node_" + std::to_string(skin.joints[j])] = j;
                }
                else
                {
                    mesh.skin[tinyGltfModel.nodes[skin.joints[j]].name] = j;
                }
            }

            mesh.inverseBindPoseTransforms = AccessorUtility::get<std::vector<glm::mat4>>(
                tinyGltfModel,
                tinyGltfModel.accessors[skin.inverseBindMatrices]);
        }

        for (const auto& texture : tinyGltfModel.textures)
        {
            auto image = getImageFromIndex(tinyGltfModel, texture.source);

            if (!memoryStorageService.textures.exists(filePath + ":" + image.name))
            {
                memoryStorageService.textures.add(filePath + ":" + image.name, image);
            }
        }

        for (const auto& tinyGltfMaterial : tinyGltfModel.materials)
        {
            if (memoryStorageService.materials.exists(filePath + ":" + tinyGltfMaterial.name))
            {
                continue;
            }

            std::optional<Texture*> albedoTexturePtr;
            std::optional<Texture*> metallicRoughnessTexturePtr;
            std::optional<Texture*> normalTexturePtr;
            std::optional<Texture*> occlusionTexturePtr;
            std::optional<Texture*> emissiveTexturePtr;

            if (tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index != -1)
            {
                auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index].source];
                albedoTexturePtr = memoryStorageService.textures.get(filePath + ":" + image.name);
            }

            if (tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index != -1)
            {
                auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index].source];
                metallicRoughnessTexturePtr = memoryStorageService.textures.get(filePath + ":" + image.name);
            }

            if (tinyGltfMaterial.normalTexture.index != -1)
            {
                auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.normalTexture.index].source];
                normalTexturePtr = memoryStorageService.textures.get(filePath + ":" + image.name);
            }

            if (tinyGltfMaterial.occlusionTexture.index != -1)
            {
                auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.occlusionTexture.index].source];
                occlusionTexturePtr = memoryStorageService.textures.get(filePath + ":" + image.name);
            }

            if (tinyGltfMaterial.emissiveTexture.index != -1)
            {
                auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.emissiveTexture.index].source];
                emissiveTexturePtr = memoryStorageService.textures.get(filePath + ":" + image.name);
            }

            auto material = memoryStorageService.materials.add(filePath + ":" + tinyGltfMaterial.name, Material {
                .albedo = albedoTexturePtr,
                .metallicRoughness = metallicRoughnessTexturePtr,
                .normal = normalTexturePtr,
                .occlusion = occlusionTexturePtr,
                .emissive = emissiveTexturePtr
            });

            mesh.materials.push_back(material);
        }

        for (const auto& bufferView : tinyGltfModel.bufferViews)
        {
            if (bufferView.target == 0)
            {
                logger.warn("bufferView.target is zero. skipping buffer view.");
                continue;
            }

            mesh.meshBuffers.push_back(MeshBuffer {
                .target = bufferView.target,
                .length = bufferView.byteLength,
                .offset = bufferView.byteOffset,
                .data = tinyGltfModel.buffers[bufferView.buffer].data
            });

            for (const auto& primitive : tinyGltfMesh.primitives)
            {
                for (auto& attribute : primitive.attributes)
                {
                    auto accessor = tinyGltfModel.accessors[attribute.second];
                    auto byteStride = accessor.ByteStride(tinyGltfModel.bufferViews[accessor.bufferView]);
                    auto attributeType = toAttributeType(attribute.first);

                    if (!attributeType.has_value())
                    {
                        continue;
                    }

                    auto indexAccessor = tinyGltfModel.accessors[primitive.indices];

                    mesh.meshPrimitives.push_back(MeshPrimitive {
                        .size = accessor.type != TINYGLTF_TYPE_SCALAR ? accessor.type : 1,
                        .type = accessor.type,
                        .componentType = accessor.componentType,
                        .stride = byteStride,
                        .bufferIndex = accessor.bufferView,
                        .mode = primitive.mode,
                        .elementCount = indexAccessor.count,
                        .indicesType = toVertexIndicesType(indexAccessor.componentType).value(),
                        .attributeType = attributeType,
                        .normalized = accessor.normalized,
                        .offset = reinterpret_cast<void*>(accessor.byteOffset),
                        .indicesOffset = reinterpret_cast<void*>(indexAccessor.byteOffset),
                        .material = primitive.material != -1 ?
                            memoryStorageService.materials.get(filePath + ":" + tinyGltfModel.materials[primitive.material].name) : nullptr
                    });
                }
            }
        }

        model.meshes.push_back(mesh);
    }

    return model;
}

std::optional<PrimitiveAttributeType> GLTFService::toAttributeType(const std::string& attributeName) const
{
    if (attributeName == "POSITION")
    {
        return PrimitiveAttributeType::Position;
    }
    else if (attributeName == "TEXCOORD_0")
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
    else if (attributeName == "JOINTS_0")
    {
        return PrimitiveAttributeType::Joint;
    }
    else if (attributeName == "WEIGHTS_0")
    {
        return PrimitiveAttributeType::SkinWeight;
    }

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
    auto image = model.images[index];

    return Texture {
        .name = image.name,
        .format = toTextureFormat(image.component),
        .pixelDataType = image.bits == 16 ? PixelDataType::UnsignedShort : PixelDataType::UnsignedByte,
        .width = image.width,
        .height = image.height,
        .data = image.image
    };
}

std::optional<VertexIndicesType> GLTFService::toVertexIndicesType(int componentType) const
{
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return VertexIndicesType::UnsignedByte;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return VertexIndicesType::UnsignedShort;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return VertexIndicesType::UnsignedInt;
        default:
            return {};
    }
}
