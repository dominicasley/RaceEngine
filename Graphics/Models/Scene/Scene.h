#pragma once

#include <deque>
#include <memory>

#include "Camera.h"
#include "Light.h"
#include "SceneNode.h"
#include "RenderableEntity.h"
#include "RenderableModel.h"

struct Scene
{
    // std::deque: element addresses stay stable under growth; references into these containers rely on it, so no erasing.
    std::deque<Camera> cameras;
    std::deque<Light> lights;
    std::deque<RenderableModel> models;
    std::deque<SceneNode> nodes;
};

