#include "EntityService.h"

EntityService::EntityService(
    spdlog::logger& logger): logger(logger) {
}

Entity& EntityService::createEntity()
{
    return entities.emplace_back(Entity(entities.size()));
}

Entity& EntityService::getEntity(unsigned long long entityId)
{
    return entities[entityId];
}
