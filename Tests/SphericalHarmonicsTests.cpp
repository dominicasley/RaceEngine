#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using Catch::Approx;
using raceengine::cubeFaceDirection;
using raceengine::cubeTexelSolidAngle;
using raceengine::evaluateShIrradiance;
using raceengine::fourPi;
using raceengine::halfToFloat;
using raceengine::projectCubeToIrradiance;
using raceengine::shBasis;
using raceengine::shCoefficientCount;

namespace
{

constexpr auto faceSize = 16u;

// A cube map whose every texel carries the radiance `shade(direction)` returns, laid out exactly
// as the projection expects: six spans of faceSize * faceSize RGB triples in cube layer order.
struct SyntheticCube
{
    std::array<std::vector<glm::vec3>, 6> storage;
    std::array<std::span<const glm::vec3>, 6> faces;

    template <typename Shade> explicit SyntheticCube(Shade shade)
    {
        for (auto face = 0u; face < 6u; face++)
        {
            storage[face].reserve(static_cast<size_t>(faceSize) * faceSize);

            for (auto y = 0u; y < faceSize; y++)
            {
                for (auto x = 0u; x < faceSize; x++)
                {
                    storage[face].push_back(shade(cubeFaceDirection(face, x, y, faceSize)));
                }
            }

            faces[face] = std::span<const glm::vec3>(storage[face]);
        }
    }

    [[nodiscard]] std::span<const std::span<const glm::vec3>, 6> view() const
    {
        return std::span<const std::span<const glm::vec3>, 6>(faces);
    }
};

// Directions spread over the sphere, for asserting a property everywhere rather than on an axis.
std::vector<glm::vec3> sampleDirections()
{
    std::vector<glm::vec3> directions;

    for (auto face = 0u; face < 6u; face++)
    {
        for (const auto x : {1u, 5u, 8u, 13u})
        {
            for (const auto y : {2u, 7u, 11u, 15u})
            {
                directions.push_back(cubeFaceDirection(face, x, y, faceSize));
            }
        }
    }

    return directions;
}

} // namespace

// The cube's texels tile the sphere exactly once. Everything downstream is a weighted sum over
// them, so a mapping that double-covered a seam or left a gap would bias the whole projection in
// a way no single coefficient's test would localise.
TEST_CASE("the cube's texel solid angles sum to the whole sphere", "[ibl][sh]")
{
    auto total = 0.0f;

    for (auto face = 0u; face < 6u; face++)
    {
        for (auto y = 0u; y < faceSize; y++)
        {
            for (auto x = 0u; x < faceSize; x++)
            {
                const auto solidAngle = cubeTexelSolidAngle(x, y, faceSize);
                CHECK(solidAngle > 0.0f);
                total += solidAngle;
            }
        }
    }

    CHECK(total == Approx(fourPi).epsilon(0.001));
}

// A cube face is a flat projection of a curved patch, so its corner texels subtend far less of
// the sphere than its centre ones. Weighting them equally is the mistake this states: it is
// invisible in any single frame and shows up as a permanent eight-fold pattern in the irradiance.
TEST_CASE("a corner texel subtends much less solid angle than a centre one", "[ibl][sh]")
{
    const auto centre = cubeTexelSolidAngle(faceSize / 2, faceSize / 2, faceSize);
    const auto corner = cubeTexelSolidAngle(0, 0, faceSize);

    CHECK(corner < centre * 0.4f);
}

// The one property that catches a wrong normalisation, a wrong cosine-lobe coefficient and a
// wrong solid angle at once: an environment of uniform radiance L illuminates a unit-albedo
// surface to outgoing radiance L, whichever way that surface faces. Anything off by pi, by 4pi or
// by a band coefficient fails here and nowhere else.
TEST_CASE("uniform radiance projects to that same radiance in every direction", "[ibl][sh]")
{
    const SyntheticCube cube([](const glm::vec3&) { return glm::vec3(0.5f, 1.25f, 3.0f); });
    const auto irradiance = projectCubeToIrradiance(cube.view(), faceSize);

    for (const auto& direction : sampleDirections())
    {
        const auto lit = evaluateShIrradiance(irradiance, direction);

        CHECK(lit.r == Approx(0.5f).epsilon(0.01));
        CHECK(lit.g == Approx(1.25f).epsilon(0.01));
        CHECK(lit.b == Approx(3.0f).epsilon(0.01));
    }
}

// Colour channels do not mix. A projection that summed into one accumulator and broadcast it, or
// that transposed a coefficient's components, would still pass the uniform test with a grey
// environment — so this one is deliberately not grey.
TEST_CASE("a single lit hemisphere lights the side that faces it and not the other", "[ibl][sh]")
{
    // Radiance from +Y only. The brightest response must be at +Y and the dimmest at -Y.
    const SyntheticCube cube([](const glm::vec3& direction)
                             { return direction.y > 0.0f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f); });
    const auto irradiance = projectCubeToIrradiance(cube.view(), faceSize);

    const auto up = evaluateShIrradiance(irradiance, glm::vec3(0.0f, 1.0f, 0.0f));
    const auto down = evaluateShIrradiance(irradiance, glm::vec3(0.0f, -1.0f, 0.0f));
    const auto side = evaluateShIrradiance(irradiance, glm::vec3(1.0f, 0.0f, 0.0f));

    CHECK(up.r > side.r);
    CHECK(side.r > down.r);
    // The unlit channels stay unlit: no radiance ever carried green or blue.
    CHECK(up.g == Approx(0.0f).margin(1e-5));
    CHECK(up.b == Approx(0.0f).margin(1e-5));

    // An order-2 reconstruction of a step function rings, so the shadowed side goes slightly
    // negative rather than to zero. Stated rather than clamped away, because it is why the shader
    // takes a max against zero and not a sign that the projection is wrong.
    CHECK(down.r < 0.05f);
}

// The basis the shader spells has to be the basis the projection used. Orthonormality is what
// makes the two the same function rather than two functions that happen to agree on an axis.
TEST_CASE("the basis is orthonormal over the sphere", "[ibl][sh]")
{
    std::array<std::array<float, shCoefficientCount>, shCoefficientCount> inner{};

    for (auto face = 0u; face < 6u; face++)
    {
        for (auto y = 0u; y < faceSize; y++)
        {
            for (auto x = 0u; x < faceSize; x++)
            {
                const auto basis = shBasis(cubeFaceDirection(face, x, y, faceSize));
                const auto solidAngle = cubeTexelSolidAngle(x, y, faceSize);

                for (size_t row = 0; row < shCoefficientCount; row++)
                {
                    for (size_t column = 0; column < shCoefficientCount; column++)
                    {
                        inner[row][column] += basis[row] * basis[column] * solidAngle;
                    }
                }
            }
        }
    }

    for (size_t row = 0; row < shCoefficientCount; row++)
    {
        for (size_t column = 0; column < shCoefficientCount; column++)
        {
            CHECK(inner[row][column] == Approx(row == column ? 1.0f : 0.0f).margin(0.02));
        }
    }
}

// The capture is R16G16B16A16_SFLOAT and the projection is CPU-side, so every probe's irradiance
// goes through this. The cases are the ones a shortened decoder gets wrong.
TEST_CASE("half-precision decoding covers the whole format", "[ibl][sh]")
{
    CHECK(halfToFloat(0x0000) == Approx(0.0f));
    CHECK(halfToFloat(0x8000) == Approx(0.0f));
    CHECK(halfToFloat(0x3C00) == Approx(1.0f));
    CHECK(halfToFloat(0xBC00) == Approx(-1.0f));
    CHECK(halfToFloat(0x4000) == Approx(2.0f));
    CHECK(halfToFloat(0x3555) == Approx(0.333251953125f));
    // Largest normal, and the smallest normal.
    CHECK(halfToFloat(0x7BFF) == Approx(65504.0f));
    CHECK(halfToFloat(0x0400) == Approx(6.103515625e-05f));
    // Subnormals: the branch that has to renormalise rather than shift.
    CHECK(halfToFloat(0x0001) == Approx(5.960464477539063e-08f));
    CHECK(halfToFloat(0x03FF) == Approx(6.097555160522461e-05f));
    // Infinity and NaN survive as themselves rather than as a huge finite number.
    CHECK(std::isinf(halfToFloat(0x7C00)));
    CHECK(std::isnan(halfToFloat(0x7E00)));
}
