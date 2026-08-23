module;

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.graphics.models:Material;

import raceengine.resource;
import :Shader;
import :Texture;

namespace raceengine
{

// How many detail layers a blended material may carry. Four because the mask that weights them is
// one RGBA texture and a channel is what selects a layer — the arrangement every terrain splat map
// has used for twenty years, and the one imported content is authored against. It is not a limit
// anybody chose independently of that: widening it means a second mask, which is a different
// feature and not a bigger number here.
export inline constexpr uint32_t detailLayerCount = 4;

// A surface whose colour comes from more than one texture: a low-resolution base that says where
// the surface changes, and detail layers tiled across it that say what it is made of.
//
// This is the classic answer to two problems at once, and it is worth being clear which. The first
// is **resolution**: a base map stretched over a kilometre of road is a handful of texels per metre
// however large the file is, and no amount of source resolution fixes it. A detail map tiled every
// metre or two is arbitrarily sharp underfoot at a fixed cost. The second is **variety**: a mask
// lets one material be tarmac here, worn grass there, and a blend of both across the join, without
// authoring a unique texture for the join.
//
// The layers are combined as a **weighted sum**, each by its own channel of the mask, and the
// result *modulates* the base colour rather than replacing it — so the base carries the large-scale
// colour and the layers carry the texture. A detail map for this arrangement is authored around mid
// grey, since a layer at exactly 1.0 leaves the base as it was.
export struct DetailLayer
{
    // How many times this layer repeats per unit of the **model's own** coordinate system, across
    // its x and z axes.
    //
    // Model space rather than UV, because that is the whole point: a road built from thirty meshes
    // has thirty unrelated UV layouts and a detail tiled in UV would change size at every seam,
    // where one tiled in space is continuous across all of them. Model space rather than *world*
    // space, because the tiling is a property of the asset — stated in whatever units the asset was
    // authored in — and a level that places or scales the model must not thereby change how coarse
    // its tarmac looks.
    float tiling = 1.0f;
};

export struct MaterialBlend
{
    std::array<DetailLayer, detailLayerCount> layers{};
    // Scales the combined layers before they modulate the base colour. The authoring knob for "how
    // much of this surface is the detail and how much is the base map", and the place a source
    // engine's own overall brightness for the blend lands.
    float strength = 1.0f;
};

// The classic reflectance model — ambient, diffuse and specular coefficients with a specular
// exponent — for content authored against it rather than against metalness-roughness.
//
// It is not a legacy fallback and not an approximation of the PBR block beside it: a material that
// states these was *tuned* by an artist looking at this exact formula, and the tuning is the asset.
// Converting it to metalness-roughness means inventing the two numbers that model has and this one
// does not (roughness is derivable from the exponent, metalness is not derivable from anything),
// and inventing them is what makes an imported track read as somebody's guess at the track.
//
// `specularExponent` is the Blinn-Phong power, not a roughness: it runs from about 1 (a surface
// with no highlight to speak of) to a few hundred (polished). A material that carries a specular
// map states the per-pixel multipliers for the first and last of these in that map's red and green
// channels — see `metallicRoughness`, which is the slot such a map rides in, and the shader that
// reads it.
// `ambient` and `diffuse` are **relative**, and that is the whole of what makes this model
// portable. A legacy asset states them as coefficients against *its own engine's* ambient constant
// and sun — two numbers that live in that engine's weather rather than in the asset — so a renderer
// with its own lighting cannot use them as multipliers without asserting that the two rigs agree in
// absolute terms. What an asset can state, and what `GLTFService` divides by on the way in, is its
// own *ordinary* material; so 1.0 here means "an ordinary surface taking the ordinary amount of
// light", which this renderer places from what it measures — the light probes for ambient, a plain
// Lambertian `1/pi` for diffuse. A material at 2.0 takes twice the ordinary amount, whatever the
// two engines' lamps were set to.
//
// `specular` is *not* relative: it stays the asset's own number, because there is no ordinary
// specular to be relative to — most materials on content like this carry no highlight at all.
//
// The defaults are therefore a neutral matte surface — the ordinary amount of both, no highlight —
// rather than zeroes, and that is load-bearing: the backend uploads them for a material that states
// none, so a shader written for this model needs no branch and no second statement of what
// "unstated" looks like. The flag beside them in the block says whether they *were* stated, which
// is a different question and is asked by the one thing that has to know.
export struct BlinnPhongShading
{
    float ambient = 1.0f;
    float diffuse = 1.0f;
    float specular = 0.0f;
    float specularExponent = 1.0f;
};

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
    // Set only where the asset states them. Absent is the ordinary case and means the metalness and
    // roughness above are the whole of what this surface reflects; present means the two models are
    // both on the material and a shader picks the one it was written for, which is why this is an
    // optional rather than a defaulted block — "stated" and "left at its defaults" are different
    // facts and only the first should make a Blinn-Phong shader shade.
    std::optional<BlinnPhongShading> blinnPhong{};
    // Set only where the asset states detail layers. Absent means the base colour is the whole of
    // this surface's colour, which is every ordinary material.
    std::optional<MaterialBlend> blend{};
    std::optional<Resource<Shader>> shader{};
    std::optional<Resource<Texture>> albedo;
    std::optional<Resource<Texture>> metallicRoughness;
    std::optional<Resource<Texture>> normal;
    std::optional<Resource<Texture>> occlusion;
    std::optional<Resource<Texture>> emissive;
    std::optional<Resource<CubeMap>> environment{};
    // Which layer applies where: one channel per layer, sampled in UV space because *where* a
    // surface changes is a property of that surface's own parametrisation, even though *what* it
    // changes to is tiled in space. A blended material with no mask weights nothing and draws as if
    // it had no layers.
    std::optional<Resource<Texture>> blendMask{};
    std::array<std::optional<Resource<Texture>>, detailLayerCount> detail{};
    std::vector<Resource<Texture>> textures{};
};

} // namespace raceengine
