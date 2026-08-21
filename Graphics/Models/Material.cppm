module;

#include <optional>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.graphics.models:Material;

import raceengine.resource;
import :Shader;
import :Texture;

namespace raceengine
{

export struct Material
{
    glm::vec4 baseColour;
    float metalness;
    float roughness;
    bool opaque = true;
    // glTF MASK: fragments under this coverage are discarded rather than blended. Zero is no test
    // at all, which is every OPAQUE and BLEND material; a cut-out is opaque geometry with holes,
    // so it writes depth, casts shadows and occludes exactly as far as its own coverage says.
    float alphaCutoff = 0.0f;
    // glTF's own flag, at last carried: a foliage card is authored to be seen from both sides, and
    // an opaque material that hid it behind back-face culling would delete half of every tree.
    bool doubleSided = false;
    // KHR_texture_transform: homogeneous 2D UV transform, applied as (transform * vec3(uv, 1)).xy.
    glm::mat3 transform = glm::mat3(1.0f);
    std::optional<Resource<Shader>> shader{};
    std::optional<Resource<Texture>> albedo;
    std::optional<Resource<Texture>> metallicRoughness;
    std::optional<Resource<Texture>> normal;
    std::optional<Resource<Texture>> occlusion;
    std::optional<Resource<Texture>> emissive;
    std::optional<Resource<CubeMap>> environment{};
    std::vector<Resource<Texture>> textures{};
};

} // namespace raceengine
