#pragma once

#include <typeindex>
#include <map>

#include "Component.h"

class Entity {
public:
    const unsigned long long id;
    std::map<std::type_index, std::shared_ptr<Component>> components;

    explicit Entity(unsigned long long id) : id(id) {};
    Entity(const Entity&) = delete;
    Entity(Entity&&) = default;
};