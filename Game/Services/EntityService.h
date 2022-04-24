#pragma once

#include <array>
#include <spdlog/logger.h>

#include "Game/Models/Entity.h"

class EntityService
{
private:
    spdlog::logger& logger;
    std::vector<Entity> entities;

public:
    explicit EntityService(spdlog::logger& logger);

    Entity& createEntity();
    [[nodiscard]] Entity& getEntity(unsigned long long entityId);

    template<typename T>
    std::shared_ptr<T> addComponent(unsigned long long entityId)
    {
        return addComponent<T>(getEntity(entityId));
    }

    template<typename T>
    std::shared_ptr<T> addComponent(Entity& entity)
    {
        auto component = std::make_shared<T>();
        entity.components[std::type_index(typeid(*component))] = component;

        return component;
    }

    template<typename T>
    void removeComponent(unsigned long long entityId)
    {
        removeComponent<T>(getEntity(entityId));
    }

    template<typename T>
    void removeComponent(Entity& entity)
    {
        entity.components.erase(std::type_index(typeid(T)));
    }

    template<typename T>
    [[nodiscard]] std::shared_ptr<T> getComponent(unsigned long long entityId)
    {
        return getComponent<T>(getEntity(entityId));
    }

    template<typename T>
    [[nodiscard]] std::shared_ptr<T> getComponent(Entity& entity)
    {
        return static_pointer_cast<T>(entity.components[std::type_index(typeid(T))]);
    }
};


