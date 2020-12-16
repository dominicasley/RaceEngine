#pragma once

#include <memory>
#include <vector>
#include "Camera.h"
#include "Light.h"
#include "SceneNode.h"
#include "RenderableEntity.h"
#include "RenderableModel.h"
#include "RenderableSkybox.h"

struct Scene
{
    std::vector<std::unique_ptr<Camera>> cameras;
    std::vector<std::unique_ptr<Light>> lights;
    std::vector<std::unique_ptr<RenderableModel>> models;
    std::vector<std::unique_ptr<RenderableSkybox>> skybox;
    std::vector<std::unique_ptr<SceneNode>> nodes;
};

