#include "Scene.h"
#include "RenderableEntity.h"
#include "Light.h"
#include "Camera.h"
#include "../Api/Resources/RenderableEntityDesc.h"

void Scene::update(const float delta) const
{
    for (auto& entity : entities)
    {
        entity->update(delta);
    }
}

size_t Scene::getEntityCount() const
{
    return entities.size();
}

RenderableEntity* Scene::getEntity(const unsigned int index) const
{
    return entities[index].get();
}

Camera* Scene::getCamera(const unsigned int index) const
{
    return cameras[index].get();
}

Camera* Scene::createCamera()
{
    return cameras.emplace_back(std::make_unique<Camera>()).get();
}

RenderableEntity* Scene::createEntity(const RenderableEntityDesc& renderableEntity)
{
    return entities.emplace_back(std::make_unique<RenderableEntity>(renderableEntity)).get();
}

Light* Scene::createLight()
{
    throw;
}

std::vector<std::unique_ptr<RenderableEntity>>& Scene::getEntities() const
{
    return entities;
}

std::vector<std::unique_ptr<Camera>>& Scene::getCameras() const
{
    return cameras;
}
