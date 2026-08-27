module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module raceengine.graphics:ShadowCascades;

import :RenderContract;
import raceengine.graphics.models;

namespace raceengine
{

// The arithmetic a cascaded shadow map is made of, and nothing else: no storage, no device, no
// service. Every function here is a pure function of its arguments, which is what lets the split
// scheme, the slice fit and the texel snap be tested without a GPU — and those three are exactly
// where an off-by-one hides, because a wrong answer still renders a plausible picture.

// Where one cascade ends, as a distance along the view axis.
//
// The practical scheme (Zhang et al.), which is a blend of the two schemes that are each wrong on
// their own. A uniform split spends most of its texels on the far half of the frustum, where a
// texel already covers metres; a logarithmic split spends them on the first few units in front of
// the eye, where nothing is ever drawn. `lambda` is the mix: 0 is uniform, 1 is logarithmic, and
// 0.5 is the value that keeps the nearest cascade sub-metre per texel without collapsing the last
// one. Element 0 is the near plane and element shadowCascadeCount is `distance`, so the returned
// array reads as the boundaries of the slices rather than as the slices themselves.
export [[nodiscard]] inline std::array<float, shadowCascadeCount + 1>
cascadeSplitDistances(const float nearPlane, const float distance, const float lambda)
{
    std::array<float, shadowCascadeCount + 1> splits{};
    // A near plane at or below zero has no logarithm; the ratio the scheme is built on does not
    // exist, so the uniform arm is the whole answer there.
    const auto near = std::max(nearPlane, 0.0001f);
    const auto far = std::max(distance, near + 0.0001f);
    const auto ratio = far / near;

    splits[0] = near;
    splits[shadowCascadeCount] = far;

    for (auto index = 1u; index < shadowCascadeCount; index++)
    {
        const auto fraction = static_cast<float>(index) / static_cast<float>(shadowCascadeCount);
        const auto logarithmic = near * std::pow(ratio, fraction);
        const auto uniform = near + (far - near) * fraction;
        splits[index] = lambda * logarithmic + (1.0f - lambda) * uniform;
    }

    return splits;
}

// The sphere that encloses one slice of a perspective frustum, in view space: how far along the
// view axis its centre sits, and its radius.
//
// A sphere rather than the slice's own bounding box, because the radius depends only on the field
// of view, the aspect ratio and the two distances — never on where the camera is or which way it
// is pointing. That is half of stabilisation: the cascade's volume is then the same size every
// frame, so a camera that only rotates cannot make the shadow edges swim.
export struct SliceSphere
{
    float centreDistance;
    float radius;
};

export [[nodiscard]] inline SliceSphere cascadeSliceSphere(const float verticalFieldOfViewDegrees,
                                                           const float aspectRatio, const float nearDistance,
                                                           const float farDistance)
{
    const auto halfHeight = std::tan(glm::radians(verticalFieldOfViewDegrees) * 0.5f);
    // The squared radial extent of the frustum per unit of depth: a corner at depth d is
    // sqrt(k2) * d from the axis.
    const auto k2 = halfHeight * halfHeight * (1.0f + aspectRatio * aspectRatio);

    // A slice wide enough for its far circle to swallow its near corners: the minimal sphere is
    // then the circumscribed circle of the far face, centred on it. Solving for equal distances
    // instead would put the centre *behind* the far plane and inflate the radius by a third at
    // this field of view.
    if (k2 * (farDistance + nearDistance) >= farDistance - nearDistance)
    {
        return SliceSphere{.centreDistance = farDistance, .radius = farDistance * std::sqrt(k2)};
    }

    // Otherwise the centre is where the near and far corners are equidistant.
    const auto centreDistance = 0.5f * (farDistance + nearDistance) * (1.0f + k2);
    const auto axial = farDistance - centreDistance;

    return SliceSphere{.centreDistance = centreDistance,
                       .radius = std::sqrt(axial * axial + farDistance * farDistance * k2)};
}

// A cascade's camera, fitted to a frustum slice and snapped so it cannot crawl.
export struct CascadeFit
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 roll;
    OrthographicVolume volume;
    float nearPlane;
    float farPlane;
    // World units one shadow-map texel covers. The whole bias budget is expressed in these, so
    // the shader needs no per-cascade tuning.
    float texelWorldSize;
    // Normalised depth per world unit along the light axis: what turns a bias in world units into
    // one the depth comparison can subtract.
    float depthPerWorldUnit;
};

// The second half of stabilisation. The volume is already a constant size (the sphere above), so
// all that is left is to stop it sliding by fractions of a texel: the sphere's centre is moved to
// the nearest whole-texel position *in the light's own frame*, whose axes never move because they
// come from the light direction alone. Snapping in world space or in the camera's frame would not
// work — the lattice has to be the one the shadow map's texels are laid out on.
//
// `casterExtent` is how far behind the slice the volume still catches geometry: a caster between
// the light and the slice is outside the sphere and must still be in the depth map, or a building
// stops casting the moment the camera looks past its base.
// `padWorld` inflates the fitted square by that many world units on every side, so a held fit
// stays valid while the ideal centre drifts up to the pad — the currency of the far-cascade
// cache. Everything below derives from the padded radius, including the snap lattice, so a
// padded fit is exactly the fit of a larger slice rather than a patched one. Zero is the
// un-padded fit every existing caller gets.
export [[nodiscard]] inline CascadeFit fitCascade(const glm::vec3& viewPosition, const glm::vec3& viewDirection,
                                                  const float verticalFieldOfViewDegrees, const float aspectRatio,
                                                  const float nearDistance, const float farDistance,
                                                  const glm::vec3& lightDirection, const uint32_t resolution,
                                                  const float casterExtent, const float padWorld = 0.0f)
{
    auto sphere = cascadeSliceSphere(verticalFieldOfViewDegrees, aspectRatio, nearDistance, farDistance);
    sphere.radius += padWorld;
    const auto centre = viewPosition + glm::normalize(viewDirection) * sphere.centreDistance;
    const auto light = glm::normalize(lightDirection);

    // Any fixed vector not parallel to the light does; it must not depend on the camera, or the
    // light's frame would rotate with the view and the snap lattice with it.
    const auto up = std::abs(light.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);

    // glm::lookAt's basis for an eye at the origin, written out because the snap needs the axes
    // themselves and not the matrix: z away from the direction of travel, x to the right, y
    // completing the frame.
    const auto axisZ = -light;
    const auto axisX = glm::normalize(glm::cross(up, axisZ));
    const auto axisY = glm::cross(axisZ, axisX);

    const auto extent = 2.0f * sphere.radius;
    const auto texelWorldSize = extent / static_cast<float>(std::max(resolution, 1u));
    const auto snap = [&](const float value)
    {
        return std::floor(value / texelWorldSize) * texelWorldSize;
    };

    // Only the two lateral axes: a shift along the light's own axis changes the stored depth
    // continuously and shows as nothing, where a sub-texel lateral shift is the crawl itself.
    const auto snapped =
        snap(glm::dot(centre, axisX)) * axisX + snap(glm::dot(centre, axisY)) * axisY + glm::dot(centre, axisZ) * axisZ;

    const auto depthRange = extent + casterExtent;

    return CascadeFit{.position = snapped - light * (sphere.radius + casterExtent),
                      .direction = light,
                      .roll = up,
                      .volume = OrthographicVolume{.left = -sphere.radius,
                                                   .right = sphere.radius,
                                                   .bottom = -sphere.radius,
                                                   .top = sphere.radius},
                      .nearPlane = 0.0f,
                      .farPlane = depthRange,
                      .texelWorldSize = texelWorldSize,
                      .depthPerWorldUnit = 1.0f / depthRange};
}

// Clip space to a shadow-map lookup: xy become texture coordinates and z the depth reference the
// comparison sampler tests, both in 0..1.
//
// y is negated, for a reason that is invisible from the shader: the scene pass records with a
// negative viewport height (docs/vulkan-abi.md), so clip y = +1 lands on framebuffer row 0 — the
// row v = 0 reads under Vulkan's top-left texture origin. The 0..1 depth correction folds into the
// same matrix: after it, z' = 0.5z + 0.5w, which is exactly what the depth buffer stores. Putting
// both here is what lets the GLSL hold no convention of its own.
//
// The sign is measured, not argued: the positive y a mirrored lookup would need moves every shadow
// to the far side of its cascade. ShadowCascadesTests states the mapping directly, so the next
// reader does not have to run a frame to check it.
export [[nodiscard]] inline glm::mat4 shadowLookupCorrection()
{
    auto correction = glm::mat4(1.0f);
    correction[0][0] = 0.5f;
    correction[1][1] = -0.5f;
    correction[2][2] = 0.5f;
    correction[3][0] = 0.5f;
    correction[3][1] = 0.5f;
    correction[3][2] = 0.5f;

    return correction;
}

} // namespace raceengine
