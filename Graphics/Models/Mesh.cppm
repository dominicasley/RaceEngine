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
    // The centre of the primitive's own bounding box, in the mesh's local space, from the POSITION
    // accessor's declared min/max. It is what the draw order sorts blended geometry by, and it has
    // to be per *primitive* rather than per node: a car's glass panels are separate primitives
    // hanging off nodes that all resolve to the same origin, so a node-based key sorts 79 draws by
    // 79 copies of the same number. Zero when the accessor declares no bounds, which glTF permits
    // for everything except POSITION and which then sorts that primitive by its own origin.
    glm::vec3 boundsCentre{0.0f};
    // Half the side lengths of that same box, from the same accessor. It is what makes the box a
    // box rather than a point, and what lets a view reject a primitive it cannot see: a draw whose
    // world-space box lies wholly outside the clip volume writes no fragments (:Frustum states the
    // exactness argument). Zero means the accessor declared no bounds — which glTF permits for
    // everything except POSITION, and which must read as "no bounds to test" rather than as a box
    // of no size, or a primitive would be culled for being unmeasured. The flag lives here rather
    // than on the centre because a symmetric primitive modelled about its own origin — a wheel, a
    // windscreen — legitimately has a centre of zero.
    glm::vec3 boundsHalfExtent{0.0f};
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
