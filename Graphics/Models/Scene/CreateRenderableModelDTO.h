#pragma once

#include "SceneNode.h"
#include "Model.h"
#include "Shader.h"

struct CreateRenderableModelDTO
{
    SceneNode* node;
    Resource<Shader> shader;
    Resource<Model> model;
};
