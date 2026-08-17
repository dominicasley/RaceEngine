module;

#include <typeindex>
#include <map>
#include <memory>

export module Game:Models;

export class Component {
};

export class Entity {
public:
    const unsigned long long id;
    std::map<std::type_index, std::shared_ptr<Component>> components;

    explicit Entity(unsigned long long id) : id(id) {};
    Entity(const Entity&) = delete;
    Entity(Entity&&) noexcept = default;
};