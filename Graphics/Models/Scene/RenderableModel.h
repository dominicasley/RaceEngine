#pragma once

#include "RenderableEntity.h"
#include "Model.h"
#include "RenderableMesh.h"

struct RenderableModel : public RenderableEntity
{
    Resource<Model> model;
    std::vector<RenderableMesh> meshes;

    explicit RenderableModel(SceneNode& node, Resource<Model> model, std::vector<RenderableMesh> meshes) :
        RenderableEntity(RenderableEntityType::Mesh, node),
        model(model),
        meshes(std::move(meshes)) { }
};