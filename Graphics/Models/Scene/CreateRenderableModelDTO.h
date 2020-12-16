#pragma once

#include "SceneNode.h"
#include "Model.h"
#include "Shader.h"

struct CreateRenderableModelDTO
{
    SceneNode* node;
    Shader* shader;
    Model* model;
};
