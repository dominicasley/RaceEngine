#pragma once

#include <vector>

#include "SceneNode.h"
#include "RenderableEntityType.h"

struct RenderableEntity
{
    RenderableEntityType type;
    SceneNode& node;

    explicit RenderableEntity(RenderableEntityType type, SceneNode& node) : type(type), node(node) {}
};



