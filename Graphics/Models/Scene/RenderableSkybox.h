#pragma once

#include "RenderableEntity.h"
#include "CubeMap.h"
#include "Shader.h"

struct RenderableSkybox : RenderableEntity
{
    CubeMap* cubeMap;
    Shader* shader;
    Model* model;

    explicit RenderableSkybox(SceneNode* node, CubeMap* cubeMap, Shader* shader, Model* model) :
        cubeMap(cubeMap),
        shader(shader),
        model(model),
        RenderableEntity(RenderableEntityType::Skybox, node) { }
};