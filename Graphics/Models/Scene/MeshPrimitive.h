#pragma once

#include "PrimitiveAttributeType.h"
#include "VertexIndicesType.h"
#include "Material.h"
#include "MeshPrimitiveAttribute.h"

struct MeshPrimitive {
    int mode;
    std::optional<Resource<Material>> material;
    size_t elementCount;
    size_t byteOffset;
    int componentType;
    int meshBufferIndex;
    std::vector<MeshPrimitiveAttribute> attributes;
};