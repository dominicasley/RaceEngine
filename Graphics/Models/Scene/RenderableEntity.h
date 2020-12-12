#pragma once

#include <vector>

#include "SceneNode.h"
#include "Model.h"
#include "RenderableMesh.h"

struct RenderableEntity
{
    Model* model;
    SceneNode* node;
    std::vector<RenderableMesh> meshes;
};



