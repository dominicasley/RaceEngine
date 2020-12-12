#pragma once

#include <memory>
#include <vector>
#include "Camera.h"
#include "Light.h"
#include "SceneNode.h"
#include "RenderableEntity.h"

struct Scene
{
    std::vector<std::unique_ptr<Camera>> cameras;
    std::vector<std::unique_ptr<Light>> lights;
    std::vector<std::unique_ptr<RenderableEntity>> entities;
    std::vector<std::unique_ptr<SceneNode>> nodes;
};

