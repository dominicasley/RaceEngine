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

void GLTFService::processNode(Model& model, const tinygltf::Model& tinyGltfModel, const tinygltf::Node& node, const glm::mat4 parentTransform) const
{
    auto transform = parentTransform * ((node.translation.size() == 3 ?
                                         glm::translate(
                                             glm::mat4(1.0f),
                                             glm::vec3(node.translation[0], node.translation[1], node.translation[2])) : glm::mat4(1.0f)) *
                                        (node.rotation.size() == 4 ?
                                         glm::mat4_cast(
                                             glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2])) : glm::mat4(1.0f)) *
                                        (node.scale.size() == 3 ?
                                         glm::scale(
                                             glm::mat4(1.0f),
                                             glm::vec3(node.scale[0], node.scale[1], node.scale[2])) : glm::mat4(1.0f)));

    if (node.mesh == -1)
    {
        for (const auto& child: node.children)
        {
            processNode(model, tinyGltfModel, tinyGltfModel.nodes[child], transform);
        }

        return;
    }

    const auto& tinyGltfMesh = tinyGltfModel.meshes[node.mesh];

    Mesh mesh;
    mesh.name = tinyGltfMesh.name;
    mesh.modelMatrix = transform;

    if (node.skin != -1)
    {
        auto& skin = tinyGltfModel.skins[node.skin];

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

    for (const auto& primitive: tinyGltfMesh.primitives)
    {
        if (primitive.indices < 0)
            continue;

        auto& indexAccessor = tinyGltfModel.accessors[primitive.indices];

        auto meshPrimitive = MeshPrimitive{
            .mode = primitive.mode,
            .material = primitive.material != -1 ?
                        std::optional<Resource<Material>>(model.materials[primitive.material]) : std::nullopt,
            .elementCount = indexAccessor.count,
            .byteOffset = indexAccessor.byteOffset,
            .componentType = indexAccessor.componentType,
            .meshBufferIndex = indexAccessor.bufferView,
            .sparseAccessor = indexAccessor.sparse.isSparse
        };

        for (auto& attribute: primitive.attributes)
        {
            auto attributeType = toAttributeType(attribute.first);

            if (!attributeType.has_value())
            {
                continue;
            }

            auto accessor = tinyGltfModel.accessors[attribute.second];
            auto byteStride = accessor.ByteStride(tinyGltfModel.bufferViews[accessor.bufferView]);

            meshPrimitive.attributes.emplace_back(MeshPrimitiveAttribute{
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

        mesh.meshPrimitives.emplace_back(meshPrimitive);
    }

    model.meshes.emplace_back(memoryStorageService.meshes.add(mesh));

    for (const auto& child: node.children)
    {
        processNode(model, tinyGltfModel, tinyGltfModel.nodes[child], transform);
    }
}

unsigned long long countMeshNodes(const tinygltf::Model& tinyGltfModel, const tinygltf::Node& node, unsigned long long count)
{
    if (node.mesh == -1)
    {
        for (const auto& child: node.children)
        {
            count += countMeshNodes(tinyGltfModel, tinyGltfModel.nodes[child], count);
        }

        return count;
    }
    else
    {
        for (const auto& child: node.children)
        {
            count += countMeshNodes(tinyGltfModel, tinyGltfModel.nodes[child], count);
        }

        return ++count;
    }
}

Model GLTFService::gltfModelToInternal(const std::string& filePath, tinygltf::Model& tinyGltfModel) const
{
    logger.info("Processing model: {}", filePath);


    unsigned long long meshCount = 0;
    for (auto& sceneNode: tinyGltfModel.scenes[tinyGltfModel.defaultScene].nodes)
    {
        meshCount += countMeshNodes(tinyGltfModel, tinyGltfModel.nodes[sceneNode], 0);
    }

    Model model;
    model.meshes.reserve(meshCount);
    model.materials.reserve(tinyGltfModel.materials.size());
    model.meshBuffers.reserve(tinyGltfModel.bufferViews.size());
    model.buffers.reserve(tinyGltfModel.buffers.size());

    std::map<std::string, Resource<Texture>> textureMap;

    for (const auto& texture: tinyGltfModel.textures)
    {
        auto image = getImageFromIndex(tinyGltfModel, texture.source);
        textureMap.insert_or_assign(filePath + ":" + image.name, memoryStorageService.textures.add(image));
    }

    for (const auto& tinyGltfMaterial: tinyGltfModel.materials)
    {
        std::optional<Resource<Texture>> albedoTexturePtr;
        std::optional<Resource<Texture>> metallicRoughnessTexturePtr;
        std::optional<Resource<Texture>> normalTexturePtr;
        std::optional<Resource<Texture>> occlusionTexturePtr;
        std::optional<Resource<Texture>> emissiveTexturePtr;

        auto textureTransform = glm::mat3(1.0f);

        if (tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index != -1)
        {
            auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index].source];
            albedoTexturePtr = textureMap[filePath + ":" + image.name];

            if (tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.extensions.contains("KHR_texture_transform")) {
                auto& transform = tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.extensions.at("KHR_texture_transform");

                auto offset = glm::vec2(0.0);
                auto scale = glm::vec2(1.0);
                auto rotation = 0.0f;

                if (transform.Has("offset")) {
                    auto& offsetTransform = transform.Get("offset");

                    offset.x = static_cast<float>(offsetTransform.Get(0).GetNumberAsDouble());
                    offset.y = static_cast<float>(offsetTransform.Get(1).GetNumberAsDouble());
                }

                if (transform.Has("scale")) {
                    auto& scaleTransform = transform.Get("scale");

                    scale.x = static_cast<float>(scaleTransform.Get(0).GetNumberAsDouble());
                    scale.y = static_cast<float>(scaleTransform.Get(1).GetNumberAsDouble());
                }

                if (transform.Has("rotation")) {
                    auto& rotationTransform = transform.Get("rotation");

                    rotation = static_cast<float>(rotationTransform.GetNumberAsDouble());
                }

                auto translationMatrix = glm::mat3(1, 0, 0, 0, 1, 0, offset.x, offset.y, 1);
                auto rotationMatrix = glm::mat3(
                    glm::cos(rotation), glm::sin(rotation), 0,
                    -glm::sin(rotation), glm::cos(rotation), 0,
                    0, 0, 1
                );
                auto scaleMatrix = glm::mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, 1);

                textureTransform = translationMatrix * rotationMatrix * scaleMatrix;
            }
        }

        if (tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index != -1)
        {
            auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index].source];
            metallicRoughnessTexturePtr = textureMap[filePath + ":" + image.name];
        }

        if (tinyGltfMaterial.normalTexture.index != -1)
        {
            auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.normalTexture.index].source];
            normalTexturePtr = textureMap[filePath + ":" + image.name];
        }

        if (tinyGltfMaterial.occlusionTexture.index != -1)
        {
            auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.occlusionTexture.index].source];
            occlusionTexturePtr = textureMap[filePath + ":" + image.name];
        }

        if (tinyGltfMaterial.emissiveTexture.index != -1)
        {
            auto image = tinyGltfModel.images[tinyGltfModel.textures[tinyGltfMaterial.emissiveTexture.index].source];
            emissiveTexturePtr = textureMap[filePath + ":" + image.name];
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
            .transform = textureTransform,
            .albedo = albedoTexturePtr,
            .metallicRoughness = metallicRoughnessTexturePtr,
            .normal = normalTexturePtr,
            .occlusion = occlusionTexturePtr,
            .emissive = emissiveTexturePtr
        });

        model.materials.emplace_back(material);
    }

    for (auto& buffer : tinyGltfModel.buffers) {
        model.buffers.emplace_back(std::move(buffer.data));
    }

    for (auto& bufferView: tinyGltfModel.bufferViews)
    {
        model.meshBuffers.emplace_back(MeshBuffer{
            .target = bufferView.target,
            .length = bufferView.byteLength,
            .offset = bufferView.byteOffset,
            .stride = bufferView.byteStride,
            .bufferIndex = bufferView.buffer,
        });
    }

    for (auto& sceneNode: tinyGltfModel.scenes[tinyGltfModel.defaultScene].nodes)
    {
        processNode(model, tinyGltfModel, tinyGltfModel.nodes[sceneNode], glm::mat4(1.0));
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

Texture GLTFService::getImageFromIndex(tinygltf::Model& model, int index) const
{
    auto& image = model.images[index];

    return Texture {
        .name = image.name,
        .format = toTextureFormat(image.component),
        .pixelDataType = image.bits == 16 ? PixelDataType::UnsignedShort : PixelDataType::UnsignedByte,
        .width = static_cast<unsigned int>(image.width),
        .height = static_cast<unsigned int>(image.height),
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
