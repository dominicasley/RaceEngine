#pragma once
#include <vector>
#include "Mesh.h"

struct Model {
    int gpuResourceId;
    std::vector<Resource<Mesh>> meshes;
};
