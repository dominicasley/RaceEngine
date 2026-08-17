module;

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <typeindex>
#include <utility>

// Single interface unit: four tiny declarations and one out-of-line service; partitions
// would add build-graph nodes without isolating anything.
export module raceengine.game;

import raceengine.graphics.models;

namespace raceengine
{

export class Component
{
public:
    virtual ~Component() = default;
};

export class Entity
{
public:
    const unsigned long long id;
    std::map<std::type_index, std::shared_ptr<Component>> components;

    explicit Entity(unsigned long long id) :
        id(id) {};
    Entity(const Entity&) = delete;
    Entity(Entity&&) = default;
};

export struct Drawable : public Component
{
public:
    RenderableEntity& renderableEntity;
    std::optional<std::function<void()>> beforeDraw;
    std::optional<std::function<void()>> afterDraw;

    explicit Drawable(RenderableEntity& _renderableEntity) :
        renderableEntity(_renderableEntity)
    {
    }
};

export class EntityService
{
private:
    std::deque<Entity> entities;

public:
    Entity& createEntity();
    [[nodiscard]] Entity& getEntity(unsigned long long entityId);

    template <typename T> std::shared_ptr<T> addComponent(unsigned long long entityId)
    {
        return addComponent<T>(getEntity(entityId));
    }

    template <typename T, class... Types> std::shared_ptr<T> addComponent(Entity& entity, Types&&... args)
    {
        auto component = std::make_shared<T>(std::forward<Types>(args)...);
        entity.components[std::type_index(typeid(*component))] = component;

        return component;
    }

    template <typename T> void removeComponent(unsigned long long entityId)
    {
        removeComponent<T>(getEntity(entityId));
    }

    template <typename T> void removeComponent(Entity& entity)
    {
        entity.components.erase(std::type_index(typeid(T)));
    }

    template <typename T> [[nodiscard]] std::shared_ptr<T> getComponent(unsigned long long entityId)
    {
        return getComponent<T>(getEntity(entityId));
    }

    template <typename T> [[nodiscard]] std::shared_ptr<T> getComponent(Entity& entity)
    {
        return static_pointer_cast<T>(entity.components[std::type_index(typeid(T))]);
    }
};

} // namespace raceengine

module :private;

namespace raceengine
{

Entity& EntityService::createEntity()
{
    return entities.emplace_back(Entity(entities.size()));
}

Entity& EntityService::getEntity(unsigned long long entityId)
{
    return entities[entityId];
}

} // namespace raceengine
