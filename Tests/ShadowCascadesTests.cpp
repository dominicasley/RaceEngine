#include <array>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

import raceengine;

using Catch::Approx;
using raceengine::cascadeSliceSphere;
using raceengine::cascadeSplitDistances;
using raceengine::fitCascade;
using raceengine::GraphicsApi;
using raceengine::shadowCascadeCount;
using raceengine::shadowLookupCorrection;

namespace
{

// The eight corners of one slice of a perspective frustum, in view space: the thing the slice
// sphere has to enclose, written out here rather than derived from the sphere so the test states
// the property independently of the code under test.
std::array<glm::vec3, 8> sliceCorners(const float verticalFieldOfViewDegrees, const float aspectRatio,
                                      const float nearDistance, const float farDistance)
{
    const auto halfHeight = std::tan(glm::radians(verticalFieldOfViewDegrees) * 0.5f);
    const auto halfWidth = halfHeight * aspectRatio;

    std::array<glm::vec3, 8> corners{};
    auto index = size_t{0};
    for (const auto depth : {nearDistance, farDistance})
    {
        for (const auto y : {-1.0f, 1.0f})
        {
            for (const auto x : {-1.0f, 1.0f})
            {
                corners[index++] = glm::vec3(x * halfWidth * depth, y * halfHeight * depth, depth);
            }
        }
    }

    return corners;
}

// A lookup as the shader performs it: clip space through the correction, perspective-divided.
glm::vec3 lookup(const GraphicsApi api, const glm::vec4& clip)
{
    const auto corrected = shadowLookupCorrection(api) * clip;

    return glm::vec3(corrected) / corrected.w;
}

constexpr auto fieldOfView = 45.0f;
constexpr auto aspectRatio = 16.0f / 9.0f;

} // namespace

TEST_CASE("the split boundaries run from the near plane to the shadow distance", "[shadow][cascades]")
{
    const auto splits = cascadeSplitDistances(1.0f, 500.0f, 0.5f);

    REQUIRE(splits.size() == shadowCascadeCount + 1);
    CHECK(splits.front() == Approx(1.0f));
    CHECK(splits.back() == Approx(500.0f));

    for (auto index = size_t{1}; index < splits.size(); index++)
    {
        CHECK(splits[index] > splits[index - 1]);
    }
}

// The two arms the practical scheme mixes, each recovered by driving lambda to its end. A test of
// the mix rather than of the endpoints: an implementation that silently ignored lambda would still
// produce an increasing sequence between the same two boundaries.
TEST_CASE("lambda 0 is the uniform split and lambda 1 the logarithmic one", "[shadow][cascades]")
{
    const auto near = 1.0f;
    const auto far = 500.0f;
    const auto uniform = cascadeSplitDistances(near, far, 0.0f);
    const auto logarithmic = cascadeSplitDistances(near, far, 1.0f);

    for (auto index = 1u; index < shadowCascadeCount; index++)
    {
        const auto fraction = static_cast<float>(index) / static_cast<float>(shadowCascadeCount);
        CHECK(uniform[index] == Approx(near + (far - near) * fraction));
        CHECK(logarithmic[index] == Approx(near * std::pow(far / near, fraction)));
        // The logarithmic arm is what keeps the nearest cascade sub-metre per texel: it spends its
        // boundaries early, so every interior one sits nearer than the uniform arm's.
        CHECK(logarithmic[index] < uniform[index]);
    }
}

// A near plane at or below zero has no logarithm, so the scheme's ratio does not exist. The
// answer has to be a usable split set rather than a NaN that propagates into every matrix.
TEST_CASE("a near plane at zero still produces an increasing split set", "[shadow][cascades]")
{
    const auto splits = cascadeSplitDistances(0.0f, 100.0f, 0.5f);

    for (auto index = size_t{1}; index < splits.size(); index++)
    {
        CHECK(std::isfinite(splits[index]));
        CHECK(splits[index] > splits[index - 1]);
    }
}

TEST_CASE("the slice sphere encloses every corner of its slice", "[shadow][cascades]")
{
    // Both arms of the fit: a short slice near the eye, where the far face's circumscribed circle
    // swallows the near corners, and a long one where the centre has to move back between them.
    for (const auto [nearDistance, farDistance] : {std::pair{1.0f, 5.0f}, std::pair{50.0f, 500.0f}})
    {
        const auto sphere = cascadeSliceSphere(fieldOfView, aspectRatio, nearDistance, farDistance);
        const auto centre = glm::vec3(0.0f, 0.0f, sphere.centreDistance);

        for (const auto& corner : sliceCorners(fieldOfView, aspectRatio, nearDistance, farDistance))
        {
            CHECK(glm::length(corner - centre) <= sphere.radius * 1.0001f);
        }
    }
}

// Half of stabilisation: the radius depends on the field of view, the aspect ratio and the two
// distances alone. A sphere that changed size as the camera turned would resize the cascade's
// volume every frame, and the shadow edges would swim.
TEST_CASE("the slice sphere does not depend on where the camera is or which way it points",
          "[shadow][cascades][stability]")
{
    const auto reference = cascadeSliceSphere(fieldOfView, aspectRatio, 10.0f, 60.0f);

    for (const auto yaw : {0.0f, 0.3f, 1.7f, 3.0f})
    {
        const auto direction = glm::vec3(std::sin(yaw), 0.2f, -std::cos(yaw));
        const auto fit = fitCascade(glm::vec3(12.0f, 3.0f, -40.0f), direction, fieldOfView, aspectRatio, 10.0f, 60.0f,
                                    glm::vec3(-0.4f, -1.0f, -0.3f), 2048u, 100.0f);

        CHECK(fit.volume.right - fit.volume.left == Approx(2.0f * reference.radius));
        CHECK(fit.volume.top - fit.volume.bottom == Approx(2.0f * reference.radius));
        CHECK(fit.texelWorldSize == Approx(2.0f * reference.radius / 2048.0f));
    }
}

// The other half: the volume's centre sits on a whole-texel position of the light's own lattice,
// whose axes come from the light direction alone. Snapping in world space or in the camera's frame
// would not stop the crawl, because the map's texels are not laid out on either.
TEST_CASE("the cascade's centre is snapped to a whole texel of the light's lattice", "[shadow][cascades][stability]")
{
    const auto lightDirection = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f));
    constexpr auto resolution = 1024u;
    constexpr auto casterExtent = 250.0f;

    const auto up = glm::vec3(0.0f, 1.0f, 0.0f);
    const auto axisZ = -lightDirection;
    const auto axisX = glm::normalize(glm::cross(up, axisZ));
    const auto axisY = glm::cross(axisZ, axisX);

    // Nudged by a fraction of a texel each time: the snapped centre must land on the same lattice
    // point for every one of them, which is the crawl this exists to remove.
    for (const auto nudge : {0.0f, 0.05f, 0.11f, 0.19f})
    {
        const auto fit = fitCascade(glm::vec3(5.0f + nudge, 2.0f, 9.0f), glm::vec3(0.0f, -0.2f, -1.0f), fieldOfView,
                                    aspectRatio, 1.0f, 40.0f, lightDirection, resolution, casterExtent);

        // The volume's centre, recovered from the position the fit pulled back along the light.
        const auto centre = fit.position + lightDirection * (fit.volume.right + casterExtent);
        const auto lateralX = glm::dot(centre, axisX) / fit.texelWorldSize;
        const auto lateralY = glm::dot(centre, axisY) / fit.texelWorldSize;

        CHECK(lateralX == Approx(std::round(lateralX)).margin(0.001));
        CHECK(lateralY == Approx(std::round(lateralY)).margin(0.001));
    }
}

// The bias budget is expressed in texels and converted by these two, so a cascade needs no tuning
// of its own. casterExtent is depth the volume gains behind the slice and nothing else: it must not
// widen the volume, or the near cascade would lose the resolution it exists for.
TEST_CASE("the caster extent lengthens the cascade's depth range without widening it", "[shadow][cascades]")
{
    const auto fitWithout = fitCascade(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), fieldOfView, aspectRatio, 1.0f,
                                       40.0f, glm::vec3(0.0f, -1.0f, 0.0f), 2048u, 0.0f);
    const auto fitWith = fitCascade(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), fieldOfView, aspectRatio, 1.0f,
                                    40.0f, glm::vec3(0.0f, -1.0f, 0.0f), 2048u, 300.0f);

    CHECK(fitWith.volume.right == Approx(fitWithout.volume.right));
    CHECK(fitWith.texelWorldSize == Approx(fitWithout.texelWorldSize));
    CHECK(fitWith.farPlane == Approx(fitWithout.farPlane + 300.0f));
    CHECK(fitWith.nearPlane == Approx(0.0f));
    CHECK(fitWith.depthPerWorldUnit == Approx(1.0f / fitWith.farPlane));
}

// Both dialects run one GLSL expression, so the whole of the difference between them lives in this
// matrix. z is the reference the comparison sampler tests and must span 0..1 on both: GL's depth
// buffer stores window z from an NDC z in -1..1, and Vulkan's stores the 0..1 the clip correction
// already produced.
TEST_CASE("the lookup correction maps clip depth to 0..1 on both backends", "[shadow][cascades][parity]")
{
    for (const auto api : {GraphicsApi::OpenGL, GraphicsApi::Vulkan})
    {
        CHECK(lookup(api, glm::vec4(0.0f, 0.0f, -1.0f, 1.0f)).z == Approx(0.0f));
        CHECK(lookup(api, glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)).z == Approx(1.0f));
        // Under a perspective divide as well: the correction is a post-multiply on the clip-space
        // side, so a w of anything but 1 must come out the same.
        CHECK(lookup(api, glm::vec4(0.0f, 0.0f, 3.0f, 6.0f)).z == Approx(0.75f));
    }
}

// The one thing that differs, and the reason it does. GL rasterises the cascade with a normal
// viewport into a bottom-left-origin texture, so clip y = -1 is the row v = 0 reads. Vulkan
// rasterises it with a *negative* viewport height (docs/vulkan-abi.md), which puts clip y = +1 at
// framebuffer row 0 — the row v = 0 reads under Vulkan's top-left texture origin. Getting this
// backwards mirrors every shadow about the middle of its cascade, which the parity gate measures.
TEST_CASE("the lookup correction flips y for Vulkan and not for OpenGL", "[shadow][cascades][parity]")
{
    CHECK(lookup(GraphicsApi::OpenGL, glm::vec4(0.0f, -1.0f, 0.0f, 1.0f)).y == Approx(0.0f));
    CHECK(lookup(GraphicsApi::OpenGL, glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)).y == Approx(1.0f));

    CHECK(lookup(GraphicsApi::Vulkan, glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)).y == Approx(0.0f));
    CHECK(lookup(GraphicsApi::Vulkan, glm::vec4(0.0f, -1.0f, 0.0f, 1.0f)).y == Approx(1.0f));

    // x is a texture coordinate on both and is never flipped: a backend that mirrored it would
    // still land inside the map, which is what makes it worth stating.
    for (const auto api : {GraphicsApi::OpenGL, GraphicsApi::Vulkan})
    {
        CHECK(lookup(api, glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f)).x == Approx(0.0f));
        CHECK(lookup(api, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)).x == Approx(1.0f));
    }
}
