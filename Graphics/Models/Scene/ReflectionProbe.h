#pragma once

#include <optional>

#include "SceneNode.h"
#include "Texture.h"
#include "Shader.h"

class ReflectionProbe {
    bool active;
    SceneNode node;
    std::optional<Resource<Texture>>
};