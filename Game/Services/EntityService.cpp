#include "EntityService.h"

EntityService::EntityService(
    spdlog::logger& logger): logger(logger) {
    entities.reserve(1024);
}

Entity& EntityService::createEntity()
{
    return entities.emplace_back(Entity(entities.size()));
}

Entity& EntityService::getEntity(unsigned long long entityId)
{
    return entities[entityId];
}
