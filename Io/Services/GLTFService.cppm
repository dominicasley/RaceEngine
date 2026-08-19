module;

#include <expected>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <spdlog/logger.h>
#include <tiny_gltf.h>

export module raceengine.io:GLTFService;

import raceengine.io.accessor;
import raceengine.graphics.models;
import raceengine.shared;

// Defined in Io/ThirdPartyImpl.cpp; see the note there for why the loader call
// cannot live in this unit.
extern "C++" bool raceengineLoadTinyGltf(tinygltf::Model& model, std::string& error, std::string& warning,
                                         const std::string& filePath, bool binary);

namespace raceengine
{

export class GLTFService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;

public:
    explicit GLTFService(spdlog::logger& logger, MemoryStorageService& memoryStorageService);
    // These three report rather than returning an empty optional or a half-built Model: the file
    // says what is wrong with it, and the sentence is the only thing that tells a caller whether
    // the asset is missing, the wrong format, or malformed in a specific place.
    [[nodiscard]] std::expected<Model, std::string> gltfModelToInternal(const std::string& fileName,
                                                                        const tinygltf::Model& tinyGltfModel) const;
    [[nodiscard]] std::expected<Model, std::string> loadModelFromFile(const std::string& filePath) const;
    [[nodiscard]] std::optional<PrimitiveAttributeType> toAttributeType(const std::string& attributeName) const;
    [[nodiscard]] TextureFormat toTextureFormat(int format) const;
    [[nodiscard]] glm::mat3 toTextureTransform(const tinygltf::TextureInfo& textureInfo) const;
    [[nodiscard]] std::expected<Texture, std::string> getImageFromIndex(const tinygltf::Model& model, int index,
                                                                        ColourSpace colourSpace) const;
    [[nodiscard]] std::optional<VertexIndicesType> toVertexIndicesType(int componentType) const;
    [[nodiscard]] std::expected<void, std::string> processNode(Model& model, const tinygltf::Model& tinyGltfModel,
                                                               const tinygltf::Node& node,
                                                               const glm::mat4 parentTransform) const;
};

GLTFService::GLTFService(spdlog::logger& logger, MemoryStorageService& memoryStorageService) :
    logger(logger),
    memoryStorageService(memoryStorageService)
{
}

std::expected<Model, std::string> GLTFService::loadModelFromFile(const std::string& filePath) const
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
        return std::unexpected("Unknown extension " + fileExtension + " when loading model with path " + filePath);
    }

    if (!result)
    {
        return std::unexpected("tinygltf could not read " + filePath + (error.empty() ? "" : ": " + error));
    }

    // tinygltf's warning is what it recovered from; its error is what it did not. A non-empty
    // error alongside a successful return used to be logged and the model used regardless, which
    // made "the file is partly unreadable" indistinguishable from "the file loaded".
    if (!error.empty())
    {
        return std::unexpected("tinygltf read " + filePath + " with an unrecovered problem: " + error);
    }

    logger.info("Loaded model: {}", filePath);

    if (!warning.empty())
    {
        logger.warn("tinygltf recovered while reading {}: {}", filePath, warning);
    }

    return gltfModelToInternal(filePath, model);
}

std::expected<void, std::string> GLTFService::processNode(Model& model, const tinygltf::Model& tinyGltfModel,
                                                          const tinygltf::Node& node,
                                                          const glm::mat4 parentTransform) const
{
    // Every index below this point comes out of the file. The walk used to index the model's
    // vectors with all of them unchecked, so a hand-edited or truncated glTF read whatever was
    // next in memory; each is range-checked now and a bad one is the file's answer, reported.
    const auto childNode = [&](const int index) -> std::expected<const tinygltf::Node*, std::string>
    {
        if (index < 0 || std::cmp_greater_equal(index, tinyGltfModel.nodes.size()))
        {
            return std::unexpected("node names child " + std::to_string(index) + " of " +
                                   std::to_string(tinyGltfModel.nodes.size()));
        }

        return &tinyGltfModel.nodes[static_cast<size_t>(index)];
    };

    // glTF states a node's transform *either* as a 4x4 matrix *or* as translation/rotation/scale,
    // never both, and an exporter picks one for the whole file: Blender writes TRS, the FBX and
    // Collada paths most car models come through write matrices. Reading only the TRS form is not a
    // partial implementation, it is a silent identity — every node collapses onto its parent, and
    // the model comes out as its parts piled at the origin at whatever spread their local spaces
    // happen to have. It reads as a wrong *scale* rather than as a wrong transform, because the
    // pile's bounding box is the one thing about it that still looks plausible.
    //
    // The matrix is column-major in the file, which is glm's own order, so the sixteen doubles map
    // straight across.
    const auto localTransform = [&]
    {
        if (node.matrix.size() == 16)
        {
            glm::mat4 matrix(1.0f);
            for (int column = 0; column < 4; column++)
            {
                for (int row = 0; row < 4; row++)
                {
                    matrix[column][row] = static_cast<float>(node.matrix[static_cast<size_t>(column * 4 + row)]);
                }
            }

            return matrix;
        }

        return (node.translation.size() == 3
                    ? glm::translate(glm::mat4(1.0f),
                                     glm::vec3(node.translation[0], node.translation[1], node.translation[2]))
                    : glm::mat4(1.0f)) *
               (node.rotation.size() == 4
                    ? glm::mat4_cast(
                          glm::quat(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                                    static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2])))
                    : glm::mat4(1.0f)) *
               (node.scale.size() == 3
                    ? glm::scale(glm::mat4(1.0f), glm::vec3(node.scale[0], node.scale[1], node.scale[2]))
                    : glm::mat4(1.0f));
    }();

    auto transform = parentTransform * localTransform;

    if (node.mesh == -1)
    {
        for (const auto& child : node.children)
        {
            const auto resolved = childNode(child);
            if (!resolved)
            {
                return std::unexpected(resolved.error());
            }

            if (const auto walked = processNode(model, tinyGltfModel, *resolved.value(), transform); !walked)
            {
                return walked;
            }
        }

        return {};
    }

    if (std::cmp_greater_equal(node.mesh, tinyGltfModel.meshes.size()))
    {
        return std::unexpected("node names mesh " + std::to_string(node.mesh) + " of " +
                               std::to_string(tinyGltfModel.meshes.size()));
    }

    const auto tinyGltfMesh = tinyGltfModel.meshes[static_cast<size_t>(node.mesh)];

    Mesh mesh;
    mesh.name = tinyGltfMesh.name;
    mesh.modelMatrix = transform;

    if (node.skin != -1)
    {
        if (std::cmp_greater_equal(node.skin, tinyGltfModel.skins.size()))
        {
            return std::unexpected("node names skin " + std::to_string(node.skin) + " of " +
                                   std::to_string(tinyGltfModel.skins.size()));
        }

        auto skin = tinyGltfModel.skins[static_cast<size_t>(node.skin)];
        for (size_t j = 0; j < skin.joints.size(); j++)
        {
            const auto jointNode = childNode(skin.joints[j]);
            if (!jointNode)
            {
                return std::unexpected("skin of mesh " + tinyGltfMesh.name + ": " + jointNode.error());
            }

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

        // inverseBindMatrices is optional (-1); the spec then defines every joint's inverse
        // bind matrix as the identity, which is what the skinning path expects to find.
        if (skin.inverseBindMatrices >= 0 && std::cmp_less(skin.inverseBindMatrices, tinyGltfModel.accessors.size()))
        {
            // The accessor reports why it could not be read now. It used to answer a malformed
            // one with an empty vector, which reads exactly like a skin that declared none — and
            // the difference is a mesh that renders in its bind pose versus one that does not
            // render right at all.
            auto inverseBindPose = AccessorUtility::get<std::vector<glm::mat4>>(
                tinyGltfModel, tinyGltfModel.accessors[static_cast<size_t>(skin.inverseBindMatrices)]);

            if (!inverseBindPose)
            {
                return std::unexpected("inverse bind matrices of mesh " + tinyGltfMesh.name + ": " +
                                       inverseBindPose.error());
            }

            mesh.inverseBindPoseTransforms = std::move(inverseBindPose).value();
        }
        else
        {
            mesh.inverseBindPoseTransforms.assign(skin.joints.size(), glm::mat4(1.0f));
        }
    }

    for (const auto& primitive : tinyGltfMesh.primitives)
    {
        if (primitive.indices < 0)
        {
            logger.warn("Skipping non-indexed primitive in mesh {}", tinyGltfMesh.name);
            continue;
        }

        if (std::cmp_greater_equal(primitive.indices, tinyGltfModel.accessors.size()))
        {
            return std::unexpected("primitive of mesh " + tinyGltfMesh.name + " names index accessor " +
                                   std::to_string(primitive.indices) + " of " +
                                   std::to_string(tinyGltfModel.accessors.size()));
        }

        auto indexAccessor = tinyGltfModel.accessors[static_cast<size_t>(primitive.indices)];

        // meshBufferIndex indexes model.meshBuffers, which is built one entry per bufferView:
        // an index accessor with no bufferView (-1, legal glTF) has no geometry to draw.
        if (indexAccessor.bufferView < 0 ||
            std::cmp_greater_equal(indexAccessor.bufferView, tinyGltfModel.bufferViews.size()))
        {
            logger.warn("Skipping primitive whose index accessor has no buffer view in mesh {}", tinyGltfMesh.name);
            continue;
        }

        if (primitive.material != -1 && std::cmp_greater_equal(primitive.material, model.materials.size()))
        {
            return std::unexpected("primitive of mesh " + tinyGltfMesh.name + " names material " +
                                   std::to_string(primitive.material) + " of " +
                                   std::to_string(model.materials.size()));
        }

        auto meshPrimitive = MeshPrimitive{
            .mode = primitive.mode == -1 ? TINYGLTF_MODE_TRIANGLES : primitive.mode,
            .material = primitive.material != -1 ? std::optional<Resource<Material>>(
                                                       model.materials[static_cast<size_t>(primitive.material)])
                                                 : std::nullopt,
            .elementCount = indexAccessor.count,
            .byteOffset = indexAccessor.byteOffset,
            .componentType = indexAccessor.componentType,
            .meshBufferIndex = indexAccessor.bufferView};

        for (auto& attribute : primitive.attributes)
        {
            auto attributeType = toAttributeType(attribute.first);

            if (!attributeType.has_value())
            {
                continue;
            }

            if (attribute.second < 0 || std::cmp_greater_equal(attribute.second, tinyGltfModel.accessors.size()))
            {
                return std::unexpected("attribute " + attribute.first + " of mesh " + tinyGltfMesh.name +
                                       " names accessor " + std::to_string(attribute.second) + " of " +
                                       std::to_string(tinyGltfModel.accessors.size()));
            }

            auto accessor = tinyGltfModel.accessors[static_cast<size_t>(attribute.second)];

            // bufferView is optional on an accessor (-1 for a sparse or zero-filled one); both
            // the stride lookup here and bufferIndex below index by it.
            if (accessor.bufferView < 0 ||
                std::cmp_greater_equal(accessor.bufferView, tinyGltfModel.bufferViews.size()))
            {
                logger.warn("Skipping attribute {} with no buffer view in mesh {}", attribute.first, tinyGltfMesh.name);
                continue;
            }

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

            // glTF requires POSITION to declare min and max, which is a bounding box for free — no
            // vertex data is read to get it. It is what orders blended geometry, and the order is
            // only as good as this is: a primitive that declares none sorts by its own origin.
            if (attributeType.value() == PrimitiveAttributeType::Position && accessor.minValues.size() == 3 &&
                accessor.maxValues.size() == 3)
            {
                meshPrimitive.boundsCentre =
                    glm::vec3(static_cast<float>(accessor.minValues[0] + accessor.maxValues[0]),
                              static_cast<float>(accessor.minValues[1] + accessor.maxValues[1]),
                              static_cast<float>(accessor.minValues[2] + accessor.maxValues[2])) *
                    0.5f;
            }
        }

        mesh.meshPrimitives.push_back(meshPrimitive);
    }

    model.meshes.push_back(memoryStorageService.meshes.add(mesh));

    for (const auto& child : node.children)
    {
        const auto resolved = childNode(child);
        if (!resolved)
        {
            return std::unexpected(resolved.error());
        }

        if (const auto walked = processNode(model, tinyGltfModel, *resolved.value(), transform); !walked)
        {
            return walked;
        }
    }

    return {};
}

std::expected<Model, std::string> GLTFService::gltfModelToInternal(const std::string& filePath,
                                                                   const tinygltf::Model& tinyGltfModel) const
{
    logger.info("Processing model: {}", filePath);
    Model model;

    std::map<int, Resource<Texture>> textureMap;

    // glTF's own division: base colour and emissive are sRGB, normal and metallic-roughness and
    // occlusion are measurements. The slot decides, but images are shared by source index and the
    // Texture has to know before it is added — mutate() is the main thread's and this runs on a
    // worker — so the colour slots are read off the materials first. An image used as both would
    // land here and be decoded, which is the right way round: the colour slot is the visible one.
    std::set<int> colourImageSources;
    for (const auto& tinyGltfMaterial : tinyGltfModel.materials)
    {
        for (const auto textureIndex :
             {tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index, tinyGltfMaterial.emissiveTexture.index})
        {
            if (textureIndex < 0 || std::cmp_greater_equal(textureIndex, tinyGltfModel.textures.size()))
            {
                continue;
            }

            const auto source = tinyGltfModel.textures[static_cast<size_t>(textureIndex)].source;
            if (source >= 0)
            {
                colourImageSources.insert(source);
            }
        }
    }

    for (const auto& texture : tinyGltfModel.textures)
    {
        if (texture.source < 0)
        {
            continue;
        }

        auto image =
            getImageFromIndex(tinyGltfModel, texture.source,
                              colourImageSources.contains(texture.source) ? ColourSpace::Srgb : ColourSpace::Linear);
        if (!image)
        {
            return std::unexpected(image.error());
        }

        textureMap.insert_or_assign(texture.source, memoryStorageService.textures.add(std::move(image).value()));
    }

    // find, not operator[]: a texture reference whose image was skipped above (source < 0) has no
    // entry, and operator[] would insert a default-constructed handle and hand it back as if it
    // named something. That used to be invisible — a default Resource passed the bounds check
    // exists() was — and is now simply absent, which is what the slot means.
    const auto textureFor = [&](const int textureIndex) -> std::optional<Resource<Texture>>
    {
        if (textureIndex < 0 || std::cmp_greater_equal(textureIndex, tinyGltfModel.textures.size()))
        {
            return std::nullopt;
        }

        const auto found = textureMap.find(tinyGltfModel.textures[static_cast<size_t>(textureIndex)].source);

        return found == textureMap.end() ? std::nullopt : std::optional<Resource<Texture>>(found->second);
    };

    for (const auto& tinyGltfMaterial : tinyGltfModel.materials)
    {
        const auto albedoTexturePtr = textureFor(tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture.index);
        const auto metallicRoughnessTexturePtr =
            textureFor(tinyGltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index);
        const auto normalTexturePtr = textureFor(tinyGltfMaterial.normalTexture.index);
        const auto occlusionTexturePtr = textureFor(tinyGltfMaterial.occlusionTexture.index);
        const auto emissiveTexturePtr = textureFor(tinyGltfMaterial.emissiveTexture.index);

        auto material = memoryStorageService.materials.add(Material{
            .baseColour = glm::vec4(tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[0],
                                    tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[1],
                                    tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[2],
                                    tinyGltfMaterial.pbrMetallicRoughness.baseColorFactor[3]),
            .metalness = static_cast<float>(tinyGltfMaterial.pbrMetallicRoughness.metallicFactor),
            .roughness = static_cast<float>(tinyGltfMaterial.pbrMetallicRoughness.roughnessFactor),
            // glTF's alphaMode, which was hardcoded true — so the one thing reading this field, the
            // back-face cull decision, always took the opaque branch and the field was a constant
            // wearing a material's name. MASK counts as not-opaque here: this engine has no alpha
            // test, so a cut-out is carried by the same blend as a pane of glass.
            .opaque = tinyGltfMaterial.alphaMode.empty() || tinyGltfMaterial.alphaMode == "OPAQUE",
            // One material-wide transform, read from the base colour reference. The extension is
            // per texture reference, so a material transforming its maps differently is flattened
            // to whatever base colour asks for.
            .transform = toTextureTransform(tinyGltfMaterial.pbrMetallicRoughness.baseColorTexture),
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
    for (const auto& tinyGltfMesh : tinyGltfModel.meshes)
    {
        for (const auto& primitive : tinyGltfMesh.primitives)
        {
            if (primitive.indices < 0)
            {
                continue;
            }

            if (std::cmp_greater_equal(primitive.indices, tinyGltfModel.accessors.size()))
            {
                return std::unexpected("primitive of mesh " + tinyGltfMesh.name + " names index accessor " +
                                       std::to_string(primitive.indices) + " of " +
                                       std::to_string(tinyGltfModel.accessors.size()));
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
            target = indexBufferViews.contains(i) ? TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER : TINYGLTF_TARGET_ARRAY_BUFFER;
        }

        if (bufferView.buffer < 0 || std::cmp_greater_equal(bufferView.buffer, tinyGltfModel.buffers.size()))
        {
            return std::unexpected("buffer view " + std::to_string(i) + " names buffer " +
                                   std::to_string(bufferView.buffer) + " of " +
                                   std::to_string(tinyGltfModel.buffers.size()));
        }

        // Store only this view's slice of the binary blob; accessor byte offsets are
        // relative to the bufferView, so they keep working against the sliced data.
        const auto& blob = tinyGltfModel.buffers[static_cast<size_t>(bufferView.buffer)].data;

        if (bufferView.byteOffset > blob.size() || bufferView.byteLength > blob.size() - bufferView.byteOffset)
        {
            return std::unexpected("buffer view " + std::to_string(i) + " spans bytes " +
                                   std::to_string(bufferView.byteOffset) + ".." +
                                   std::to_string(bufferView.byteOffset + bufferView.byteLength) + " of a " +
                                   std::to_string(blob.size()) + " byte buffer");
        }

        const auto sliceBegin = blob.begin() + static_cast<std::ptrdiff_t>(bufferView.byteOffset);

        model.meshBuffers.push_back(
            MeshBuffer{.target = target,
                       .length = bufferView.byteLength,
                       .offset = 0,
                       .stride = bufferView.byteStride,
                       .data = std::vector<unsigned char>(
                           sliceBegin, sliceBegin + static_cast<std::ptrdiff_t>(bufferView.byteLength))});
    }

    if (tinyGltfModel.scenes.empty())
    {
        logger.warn("Model {} contains no scenes; it contributes materials and buffers but no meshes", filePath);
        return model;
    }

    // defaultScene is optional (-1) and only meaningful as an index into scenes.
    const auto defaultScene = tinyGltfModel.defaultScene;
    const auto sceneIndex = defaultScene >= 0 && std::cmp_less(defaultScene, tinyGltfModel.scenes.size())
                                ? static_cast<size_t>(defaultScene)
                                : size_t{0};
    for (auto& sceneNode : tinyGltfModel.scenes[sceneIndex].nodes)
    {
        if (sceneNode < 0 || std::cmp_greater_equal(sceneNode, tinyGltfModel.nodes.size()))
        {
            return std::unexpected("scene names root node " + std::to_string(sceneNode) + " of " +
                                   std::to_string(tinyGltfModel.nodes.size()));
        }

        if (const auto walked =
                processNode(model, tinyGltfModel, tinyGltfModel.nodes[static_cast<size_t>(sceneNode)], glm::mat4(1.0));
            !walked)
        {
            return std::unexpected(filePath + ": " + walked.error());
        }
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

// KHR_texture_transform. tinygltf::Value::Get asserts on the wrong container type and yields a
// null Value (numeric 0) for an absent key, so every lookup is typed and length-checked first.
glm::mat3 GLTFService::toTextureTransform(const tinygltf::TextureInfo& textureInfo) const
{
    const auto extension = textureInfo.extensions.find("KHR_texture_transform");
    if (extension == textureInfo.extensions.end() || !extension->second.IsObject())
    {
        return glm::mat3(1.0f);
    }

    const auto& transform = extension->second;
    const auto readPair = [&transform](const std::string& key, const glm::vec2 fallback)
    {
        const auto& value = transform.Get(key);
        if (!value.IsArray() || value.ArrayLen() < 2)
        {
            return fallback;
        }

        return glm::vec2(static_cast<float>(value.Get(0).GetNumberAsDouble()),
                         static_cast<float>(value.Get(1).GetNumberAsDouble()));
    };

    const auto offset = readPair("offset", glm::vec2(0.0f));
    const auto scale = readPair("scale", glm::vec2(1.0f));
    const auto rotation = static_cast<float>(transform.Get("rotation").GetNumberAsDouble());

    // Spec order translate * rotate * scale, written as glm takes columns.
    const auto translation = glm::mat3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, offset.x, offset.y, 1.0f);
    const auto rotate = glm::mat3(glm::cos(rotation), glm::sin(rotation), 0.0f, -glm::sin(rotation), glm::cos(rotation),
                                  0.0f, 0.0f, 0.0f, 1.0f);
    const auto stretch = glm::mat3(scale.x, 0.0f, 0.0f, 0.0f, scale.y, 0.0f, 0.0f, 0.0f, 1.0f);

    return translation * rotate * stretch;
}

std::expected<Texture, std::string> GLTFService::getImageFromIndex(const tinygltf::Model& model, int index,
                                                                   const ColourSpace colourSpace) const
{
    if (index < 0 || std::cmp_greater_equal(index, model.images.size()))
    {
        return std::unexpected("texture names image " + std::to_string(index) + " of " +
                               std::to_string(model.images.size()));
    }

    auto image = model.images[static_cast<size_t>(index)];

    auto width = static_cast<unsigned int>(image.width);
    auto height = static_cast<unsigned int>(image.height);

    return Texture{.name = image.name,
                   .format = toTextureFormat(image.component),
                   .pixelDataType = image.bits == 16 ? PixelDataType::UnsignedShort : PixelDataType::UnsignedByte,
                   .colourSpace = colourSpace,
                   .width = width,
                   .height = height,
                   .data = image.image};
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
