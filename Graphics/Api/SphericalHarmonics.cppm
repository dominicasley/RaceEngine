module;

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include <glm/glm.hpp>

export module raceengine.graphics:SphericalHarmonics;

import raceengine.graphics.models;

namespace raceengine
{

// Projecting a cube map of radiance onto the order-2 spherical harmonic basis, and nothing else:
// no storage, no device, no service. Every function here is a pure function of its arguments, on
// the same terms as :ShadowCascades and for the same reason — an irradiance that is subtly wrong
// still renders a plausible picture, so the arithmetic has to be checkable without a GPU.
//
// The method is Ramamoorthi & Hanrahan's: project radiance onto the first nine basis functions,
// then multiply each band by the corresponding coefficient of the clamped-cosine lobe's own
// expansion. That product *is* the irradiance, because convolution against a rotationally
// symmetric kernel is a per-band scale in this basis. Nine coefficients hold it because the
// kernel's bands fall off fast and l = 3 vanishes outright.

export inline constexpr float fourPi = 12.566370614359172f;

// The nine basis functions evaluated in a direction, in the order the coefficients are stored.
// Real, orthonormal, and the constants are the closed forms rather than anything derived at
// runtime: they are what the shader also spells, so a difference between the two would be a
// difference between how light is projected and how it is read back.
export [[nodiscard]] inline std::array<float, shCoefficientCount> shBasis(const glm::vec3& direction)
{
    const auto x = direction.x;
    const auto y = direction.y;
    const auto z = direction.z;

    return {
        // l = 0
        0.282095f,
        // l = 1
        0.488603f * y,
        0.488603f * z,
        0.488603f * x,
        // l = 2
        1.092548f * x * y,
        1.092548f * y * z,
        0.315392f * (3.0f * z * z - 1.0f),
        1.092548f * x * z,
        0.546274f * (x * x - y * y),
    };
}

// The clamped-cosine lobe's expansion, per band, already divided by pi.
//
// Dividing here rather than in the shader is what makes the stored coefficients *outgoing diffuse
// radiance for unit albedo*: irradiance E = sum(c_k * A_l * Y_k), the Lambertian BRDF is
// albedo/pi, so the shading term is albedo * sum(c_k * (A_l / pi) * Y_k). Folding the constant in
// once at projection time leaves the shader multiplying by albedo and nothing else, and means no
// consumer can forget it.
[[nodiscard]] constexpr float cosineLobeBand(const size_t coefficient)
{
    // A_0 = pi, A_1 = 2pi/3, A_2 = pi/4, each over pi.
    if (coefficient == 0)
    {
        return 1.0f;
    }

    return coefficient < 4 ? 2.0f / 3.0f : 0.25f;
}

// The direction a cube map samples for the centre of texel (x, y) of face `face`.
//
// This is the mapping the hardware itself uses (the same table in the Vulkan and OpenGL specs, in
// the layer order +X, -X, +Y, -Y, +Z, -Z), and it has to be: these coefficients are projected from
// an image the GPU rendered and then read back by a shader sampling the same cube. A private
// convention here would be a rotation applied to the irradiance and to nothing else.
export [[nodiscard]] inline glm::vec3 cubeFaceDirection(const unsigned int face, const unsigned int x,
                                                        const unsigned int y, const unsigned int size)
{
    const auto extent = static_cast<float>(size);
    const auto s = 2.0f * ((static_cast<float>(x) + 0.5f) / extent) - 1.0f;
    const auto t = 2.0f * ((static_cast<float>(y) + 0.5f) / extent) - 1.0f;

    switch (face)
    {
    case 0:
        return glm::normalize(glm::vec3(1.0f, -t, -s));
    case 1:
        return glm::normalize(glm::vec3(-1.0f, -t, s));
    case 2:
        return glm::normalize(glm::vec3(s, 1.0f, t));
    case 3:
        return glm::normalize(glm::vec3(s, -1.0f, -t));
    case 4:
        return glm::normalize(glm::vec3(s, -t, 1.0f));
    default:
        return glm::normalize(glm::vec3(-s, -t, -1.0f));
    }
}

// The solid angle texel (x, y) of a cube face subtends at the centre.
//
// Not a constant across the face, and the difference is not small: a cube's corner texel covers
// roughly a third of what its centre texel does, because the projection from the sphere onto a
// flat face stretches towards the corners. Weighting every texel equally instead brightens the
// corners of every face, which shows up as a faint eight-fold pattern in the irradiance that no
// amount of tuning elsewhere removes. The closed form is the definite integral of the projected
// area element over the texel's footprint.
export [[nodiscard]] inline float cubeTexelSolidAngle(const unsigned int x, const unsigned int y,
                                                      const unsigned int size)
{
    const auto areaElement = [](const float u, const float v)
    {
        return std::atan2(u * v, std::sqrt(u * u + v * v + 1.0f));
    };

    const auto extent = static_cast<float>(size);
    const auto step = 2.0f / extent;
    const auto s = 2.0f * ((static_cast<float>(x) + 0.5f) / extent) - 1.0f;
    const auto t = 2.0f * ((static_cast<float>(y) + 0.5f) / extent) - 1.0f;

    const auto s0 = s - step * 0.5f;
    const auto s1 = s + step * 0.5f;
    const auto t0 = t - step * 0.5f;
    const auto t1 = t + step * 0.5f;

    return areaElement(s1, t1) - areaElement(s0, t1) - areaElement(s1, t0) + areaElement(s0, t0);
}

// Projects a cube map of radiance to the irradiance coefficients the shading side reads.
//
// `faces` is six spans of `size * size` RGB triples, in cube layer order. It is a span of spans
// rather than one contiguous block because the caller reads them out of a mapped staging buffer
// whose rows the driver may have padded, and re-packing that would be a copy for nothing.
//
// The result is normalised by the total solid angle actually accumulated rather than by 4pi. The
// two agree to within rounding for a complete cube; they do not agree if a caller ever hands over
// a partial one, and normalising by what was integrated is the answer that stays energy-correct
// when they diverge.
export [[nodiscard]] inline ShIrradiance
projectCubeToIrradiance(const std::span<const std::span<const glm::vec3>, 6> faces, const unsigned int size)
{
    std::array<glm::vec3, shCoefficientCount> accumulated{};
    auto totalSolidAngle = 0.0f;

    for (auto face = 0u; face < 6u; face++)
    {
        const auto& texels = faces[face];

        for (auto y = 0u; y < size; y++)
        {
            for (auto x = 0u; x < size; x++)
            {
                const auto index = static_cast<size_t>(y) * size + x;
                if (index >= texels.size())
                {
                    continue;
                }

                const auto solidAngle = cubeTexelSolidAngle(x, y, size);
                const auto basis = shBasis(cubeFaceDirection(face, x, y, size));
                const auto radiance = texels[index];

                totalSolidAngle += solidAngle;

                for (size_t coefficient = 0; coefficient < shCoefficientCount; coefficient++)
                {
                    accumulated[coefficient] += radiance * (basis[coefficient] * solidAngle);
                }
            }
        }
    }

    // The sum above already *is* the projection integral, texel solid angles included. What this
    // corrects is the quadrature's own error: the texel solid angles of a complete cube sum to
    // 4pi only up to rounding, and rescaling by the shortfall is what keeps a uniformly lit
    // environment reading back at exactly the radiance it was lit with.
    //
    // A cube with no texels integrates nothing; the coefficients are already zero and dividing
    // would make them not-a-number, which would then propagate into every fragment the probe
    // touches rather than showing up here.
    const auto normalisation = totalSolidAngle > 0.0f ? fourPi / totalSolidAngle : 0.0f;

    ShIrradiance irradiance{};
    for (size_t coefficient = 0; coefficient < shCoefficientCount; coefficient++)
    {
        const auto scaled = accumulated[coefficient] * (normalisation * cosineLobeBand(coefficient));
        irradiance[coefficient] = glm::vec4(scaled, 0.0f);
    }

    return irradiance;
}

// Evaluates what the shader evaluates, in C++. Not used by the frame — the shader does its own —
// but it is what lets a test say "an environment of uniform radiance L projects to an irradiance
// that reads back as L in every direction", which is the one property that catches a wrong
// normalisation, a wrong lobe coefficient and a wrong solid angle all at once.
export [[nodiscard]] inline glm::vec3 evaluateShIrradiance(const ShIrradiance& irradiance, const glm::vec3& direction)
{
    const auto basis = shBasis(direction);

    auto result = glm::vec3(0.0f);
    for (size_t coefficient = 0; coefficient < shCoefficientCount; coefficient++)
    {
        result += glm::vec3(irradiance[coefficient]) * basis[coefficient];
    }

    return result;
}

// IEEE binary16 to binary32.
//
// The probe's radiance is captured into R16G16B16A16_SFLOAT — the format the rest of the frame
// already renders in, so the capture builds no pipeline the scene pass did not already need — and
// the projection above runs on the CPU, so something has to decode it. Written out rather than
// reached for because there is no portable standard-library half, and the two special cases
// (subnormal, and infinity/NaN) are exactly the ones a shortened version gets wrong.
export [[nodiscard]] inline float halfToFloat(const uint16_t half)
{
    const auto sign = static_cast<uint32_t>(half >> 15) << 31;
    const auto exponent = static_cast<uint32_t>((half >> 10) & 0x1Fu);
    const auto mantissa = static_cast<uint32_t>(half & 0x3FFu);

    uint32_t bits = 0;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            // Signed zero.
            bits = sign;
        }
        else
        {
            // Subnormal: binary32 has the exponent range to hold it as a normal number, so the
            // mantissa is shifted up until its leading one reaches the implied-one position and
            // the exponent pays for each shift.
            //
            // The bias is 127 - 14, not the 127 - 15 a normal number uses. A half subnormal is
            // mantissa * 2^-24, and after `shift` doublings the value reads as 1.f * 2^(-14 -
            // shift) — one exponent step away from where the normal case sits, which is the whole
            // of what makes subnormals a separate branch rather than a smaller exponent.
            auto shifted = mantissa;
            auto shift = 0u;
            while ((shifted & 0x400u) == 0)
            {
                shifted <<= 1;
                shift++;
            }

            bits = sign | ((127u - 14u - shift) << 23) | ((shifted & 0x3FFu) << 13);
        }
    }
    else if (exponent == 0x1Fu)
    {
        // Infinity or NaN; the mantissa carries which, and shifting it up preserves that.
        bits = sign | 0x7F800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    return std::bit_cast<float>(bits);
}

} // namespace raceengine
