#pragma once

#include "SceneNode.h"
#include "Model.h"
#include "Shader.h"

struct RenderableEntityDesc
{
    Model* model;
    SceneNode* node;
    Shader* shader;
};
