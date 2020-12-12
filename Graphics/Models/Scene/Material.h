#pragma once

#include <glm/glm.hpp>
#include <optional>
#include "Texture.h"

struct Material {
    glm::vec4 baseColour;
    std::optional<Texture*> albedo;
    std::optional<Texture*> metallicRoughness;
    std::optional<Texture*> normal;
    std::optional<Texture*> occlusion;
    std::optional<Texture*> emissive;
};