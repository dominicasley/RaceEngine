#pragma once

#include "PrimitiveAttributeType.h"
#include "VertexIndicesType.h"
#include "Material.h"

struct MeshPrimitive {
    int size;
    int type;
    int componentType;
    int stride;
    int bufferIndex;
    int mode;
    size_t elementCount;
    VertexIndicesType indicesType;
    std::optional<PrimitiveAttributeType> attributeType;
    bool normalized;
    void* offset;
    void* indicesOffset;
    std::optional<Resource<Material>> material;
};