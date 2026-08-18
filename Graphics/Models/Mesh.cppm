module;

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.graphics.models:Mesh;

import raceengine.resource;
import :Material;

namespace raceengine
{

export enum class PrimitiveAttributeType { Position, TextureCoordinate, Normal, Tangent, Joint, SkinWeight };

export enum class VertexIndicesType { UnsignedByte, UnsignedShort, UnsignedInt };

export struct MeshBuffer
{
    int target;
    std::optional<unsigned int> gpuId{};
    size_t length;
    size_t offset;
    size_t stride;
    std::vector<unsigned char> data;
};

export struct MeshPrimitiveAttribute
{
    int size;
    int type;
    int componentType;
    int stride;
    int bufferIndex;
    std::optional<PrimitiveAttributeType> attributeType;
    bool normalized;
    size_t offset;
};

export struct MeshPrimitive
{
    int mode;
    // Each primitive owns its VAO: attribute bindings are VAO state, so primitives sharing a
    // mesh-level VAO would overwrite each other's vertex setup (they did, once).
    std::optional<unsigned int> gpuVao{};
    std::optional<Resource<Material>> material;
    size_t elementCount;
    size_t byteOffset;
    int componentType;
    int meshBufferIndex;
    std::vector<MeshPrimitiveAttribute> attributes{};
};

export struct Mesh
{
    std::string name;
    std::optional<unsigned int> gpuResourceId;
    std::vector<MeshPrimitive> meshPrimitives;
    std::map<std::string, int> skin;
    std::vector<glm::mat4> inverseBindPoseTransforms;
    glm::mat4 modelMatrix;
};

export struct Model
{
    std::vector<Resource<Mesh>> meshes;
    std::vector<MeshBuffer> meshBuffers;
    std::vector<Resource<Material>> materials;
};

} // namespace raceengine
