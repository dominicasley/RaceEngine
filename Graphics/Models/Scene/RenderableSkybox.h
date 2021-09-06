#pragma once

#include "RenderableEntity.h"
#include "CubeMap.h"
#include "Shader.h"

struct RenderableSkybox : RenderableEntity
{
    Resource<CubeMap> cubeMap;
    Resource<Shader> shader;
    Resource<Model> model;

    explicit RenderableSkybox(SceneNode* node, Resource<CubeMap> cubeMap, Resource<Shader> shader, Resource<Model> model) :
        cubeMap(cubeMap),
        shader(shader),
        model(model),
        RenderableEntity(RenderableEntityType::Skybox, node) { }
};