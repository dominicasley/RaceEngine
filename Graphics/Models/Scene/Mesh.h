#pragma once
#include <string>
#include <vector>
#include <map>
#include <optional>
#include "MeshBuffer.h"
#include "MeshPrimitive.h"

struct Mesh {
    std::string name;
    std::optional<unsigned int> gpuResourceId;
    std::vector<MeshBuffer> meshBuffers;
    std::vector<MeshPrimitive> meshPrimitives;
    std::vector<Resource<Material>> materials;
    std::map<std::string, int> skin;
    std::vector<glm::mat4> inverseBindPoseTransforms;
};
