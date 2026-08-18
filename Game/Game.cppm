module;

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

// An entity's behaviour on the fixed simulation tick: the entry point a game object carries
// its logic in. State belongs in the subclass, which the entity owns, so a behaviour outlives
// whatever builder registered it.
export class Behaviour : public Component
{
public:
    virtual void update(float delta) = 0;
};

export struct Drawable : public Component
{
public:
    RenderableEntity& renderableEntity;
    // Views onto the renderable's own hooks, not copies of them: a game assigns through the
    // component it owns, and the backend reads them off the entity it is already submitting.
    // That is what puts them around the individual draw call rather than around the whole
    // frame — the renderer never has to see a Drawable, so the module graph stays as it is.
    std::optional<std::function<void()>>& beforeDraw;
    std::optional<std::function<void()>>& afterDraw;

    explicit Drawable(RenderableEntity& _renderableEntity) :
        renderableEntity(_renderableEntity),
        beforeDraw(_renderableEntity.beforeDraw),
        afterDraw(_renderableEntity.afterDraw)
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
    void update(float delta);

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

    // Returns null when the entity carries no T. find, not operator[]: a miss through
    // operator[] inserts a null component and hands it back, so reading mutates the entity.
    template <typename T> [[nodiscard]] std::shared_ptr<T> getComponent(Entity& entity)
    {
        const auto component = entity.components.find(std::type_index(typeid(T)));
        if (component == entity.components.end())
        {
            return nullptr;
        }

        return std::static_pointer_cast<T>(component->second);
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

// Ids are dense indices handed out by createEntity and nothing is ever removed, so an id this
// service never issued is a caller bug, not a runtime condition: it is reported as one instead
// of indexing the deque out of bounds.
Entity& EntityService::getEntity(unsigned long long entityId)
{
    if (entityId >= entities.size())
    {
        throw std::out_of_range("No entity with id " + std::to_string(entityId) + "; " +
                                std::to_string(entities.size()) + " entities exist");
    }

    return entities[entityId];
}

// Components are keyed by their *dynamic* type, so a Behaviour subclass is not reachable
// through a type_index lookup for Behaviour and the sweep has to ask each component whether it
// is one. That is the cost of the map's identity model, not of the tick.
void EntityService::update(float delta)
{
    for (auto& entity : entities)
    {
        for (auto& component : entity.components)
        {
            if (auto* behaviour = dynamic_cast<Behaviour*>(component.second.get()))
            {
                behaviour->update(delta);
            }
        }
    }
}

} // namespace raceengine
