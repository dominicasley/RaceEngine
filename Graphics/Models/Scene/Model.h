#pragma once
#include <vector>
#include <Shared/Types/Resource.h>
#include "Material.h"
#include "Mesh.h"
#include "MeshBuffer.h"

struct Model {
    std::vector<Resource<Mesh>> meshes;
    std::vector<MeshBuffer> meshBuffers;
    std::vector<Resource<Material>> materials;
};
