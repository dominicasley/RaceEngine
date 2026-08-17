#pragma once
#include <vector>
#include "Mesh.h"

struct Model {
    std::vector<Resource<Mesh>> meshes;
    std::vector<MeshBuffer> meshBuffers;
    std::vector<std::vector<unsigned char>> buffers;
    std::vector<Resource<Material>> materials;
};
