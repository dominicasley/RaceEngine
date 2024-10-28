#pragma once

#include "PrimitiveAttributeType.h"
#include "VertexIndicesType.h"
#include "Material.h"

struct MeshPrimitiveAttribute {
    int size;
    int type;
    int componentType;
    int stride;
    int bufferIndex;
    std::optional<PrimitiveAttributeType> attributeType;
    bool normalized;
    size_t offset;
};