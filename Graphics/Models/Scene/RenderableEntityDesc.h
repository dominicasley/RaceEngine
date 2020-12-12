#pragma once

#include "SceneNode.h"
#include "Model.h"

struct RenderableEntityDesc
{
    Model* model;
    SceneNode* node;
};
