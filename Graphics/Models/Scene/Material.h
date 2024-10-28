#pragma once

#include <glm/glm.hpp>
#include <optional>
#include <Shared/Types/Resource.h>

#include "Texture.h"
#include "CubeMap.h"
#include "Shader.h"

struct Material
{
    glm::vec4 baseColour;
    float metalness;
    float roughness;
    bool opaque = true;
    glm::vec2 repeat = glm::vec2(1.0f);
    std::optional<Resource<Shader>> shader;
    std::optional<Resource<Texture>> albedo;
    std::optional<Resource<Texture>> metallicRoughness;
    std::optional<Resource<Texture>> normal;
    std::optional<Resource<Texture>> occlusion;
    std::optional<Resource<Texture>> emissive;
    std::optional<Resource<Texture>> environment;
    std::vector<Resource<Texture>> textures;
};