#pragma once

#include "SceneNode.h"
#include "Shader.h"
#include "CubeMap.h"
#include "Model.h"

struct CreateRenderableSkyboxDTO
{
    SceneNode* node;
    Resource<Shader> shader;
    Resource<CubeMap> cubeMap;
    Resource<Model> model;
};
