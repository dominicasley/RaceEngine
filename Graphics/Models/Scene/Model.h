#pragma once
#include <vector>
#include "Mesh.h"

struct Model {
    std::optional<int> gpuResourceId;
    std::vector<Resource<Mesh>> meshes;
};
