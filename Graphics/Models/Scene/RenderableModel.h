#pragma once

#include "RenderableEntity.h"
#include "Model.h"
#include "RenderableMesh.h"

struct RenderableModel : public RenderableEntity
{
    Model* model;
    std::vector<RenderableMesh> meshes;

    explicit RenderableModel(SceneNode* node, Model* model, std::vector<RenderableMesh> meshes) :
        model(model),
        meshes(std::move(meshes)),
        RenderableEntity(RenderableEntityType::Mesh, node) { }
};