#pragma once

#include "SceneNode.h"
#include "Shader.h"
#include "CubeMap.h"
#include "Model.h"

struct CreateRenderableSkyboxDTO
{
    SceneNode* node;
    Shader* shader;
    CubeMap* cubeMap;
    Model* model;
};
