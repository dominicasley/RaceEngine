#pragma once

#include <memory>
#include <vector>
#include "Camera.h"
#include "Light.h"
#include "RenderableEntity.h"


class Scene
{
private:
    mutable std::vector<std::unique_ptr<Camera>> cameras;
    mutable std::vector<std::unique_ptr<Light>> lights;
    mutable std::vector<std::unique_ptr<RenderableEntity>> entities;

public:
    void update(float delta) const;
    [[nodiscard]] size_t getEntityCount() const;
    [[nodiscard]] RenderableEntity* createEntity(const RenderableEntityDesc &);
    [[nodiscard]] RenderableEntity* getEntity(unsigned int index) const;
    [[nodiscard]] std::vector<std::unique_ptr<RenderableEntity>>& getEntities() const;
    [[nodiscard]] Camera* createCamera();
    [[nodiscard]] Camera* getCamera(unsigned int index) const;
    [[nodiscard]] std::vector<std::unique_ptr<Camera>>& getCameras() const;
    [[nodiscard]] Light* createLight();
};

