#include "GLTFService.h"
#include "../Utility/AccessorUtility.h"

#include <set>
#include <utility>

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

    const auto tinyGltfMesh = tinyGltfModel.meshes[node.mesh];

    Mesh mesh;
    mesh.name = tinyGltfMesh.name;
    mesh.modelMatrix = transform;

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

    for (const auto& primitive: tinyGltfMesh.primitives)
    {
        if (primitive.indices < 0)
        {
            logger.warn("Skipping non-indexed primitive in mesh {}", tinyGltfMesh.name);
            continue;
        }

        auto indexAccessor = tinyGltfModel.accessors[primitive.indices];

        auto meshPrimitive = MeshPrimitive{
            .mode = primitive.mode == -1 ? TINYGLTF_MODE_TRIANGLES : primitive.mode,
            .material = primitive.material != -1 ?
                        std::optional<Resource<Material>>(model.materials[primitive.material]) : std::nullopt,
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

            auto accessor = tinyGltfModel.accessors[attribute.second];
            auto byteStride = accessor.ByteStride(tinyGltfModel.bufferViews[accessor.bufferView]);

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
        processNode(model, tinyGltfModel, tinyGltfModel.nodes[child], transform);
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
            albedoTexturePtr = textureMap[tinyGltfModel.textures[tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index].source];
        }

        if (tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index != -1)
        {
            metallicRoughnessTexturePtr = textureMap[tinyGltfModel.textures[tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index].source];
        }

        if (tinyGltfMaterial.normalTexture.index != -1)
        {
            normalTexturePtr = textureMap[tinyGltfModel.textures[tinyGltfMaterial.normalTexture.index].source];
        }

        if (tinyGltfMaterial.occlusionTexture.index != -1)
        {
            occlusionTexturePtr = textureMap[tinyGltfModel.textures[tinyGltfMaterial.occlusionTexture.index].source];
        }

        if (tinyGltfMaterial.emissiveTexture.index != -1)
        {
            emissiveTexturePtr = textureMap[tinyGltfModel.textures[tinyGltfMaterial.emissiveTexture.index].source];
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
    for (auto& sceneNode: tinyGltfModel.scenes[sceneIndex].nodes)
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

Texture GLTFService::getImageFromIndex(const tinygltf::Model& model, int index) const
{
    auto image = model.images[index];

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
