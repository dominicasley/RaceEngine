#pragma once

#include <glm/glm.hpp>
#include <optional>
#include "Texture.h"
#include "Shader.h"

struct Material {
    glm::vec4 baseColour;
    glm::vec2 repeat = glm::vec2(1.0f);
    Shader* shader;
    std::optional<Texture*> albedo;
    std::optional<Texture*> metallicRoughness;
    std::optional<Texture*> normal;
    std::optional<Texture*> occlusion;
    std::optional<Texture*> emissive;
};