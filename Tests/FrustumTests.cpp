#include <array>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

import raceengine;

using Catch::Approx;
using raceengine::aabbOutsideFrustum;
using raceengine::fitCascade;
using raceengine::frustumPlanes;

namespace
{

// A box of no size at a point, which is how the tests below ask "is this *point* inside?" without
// a second entry point. The half-extent has to be non-zero for the renderer's own guard, but this
// function is under no such obligation — it is the test that decides what it is asking.
[[nodiscard]] bool pointOutside(const std::array<glm::vec4, 6>& planes, const glm::vec3& point)
{
    return aabbOutsideFrustum(planes, glm::mat4(1.0f), point, glm::vec3(0.0f), 0.0f);
}

[[nodiscard]] bool boxOutside(const std::array<glm::vec4, 6>& planes, const glm::vec3& centre,
                              const glm::vec3& halfExtent, const glm::mat4& localToWorld = glm::mat4(1.0f))
{
    return aabbOutsideFrustum(planes, localToWorld, centre, halfExtent, 0.0f);
}

} // namespace

// The engine keeps glm's GL depth convention for both projection arms and applies Vulkan's 0..1
// correction downstream, so the extraction is the textbook one. If CameraService ever stops doing
// that, this is the test that says so.
TEST_CASE("frustum planes bound a perspective volume", "[graphics][frustum]")
{
    const auto projection = glm::perspective(glm::radians(75.0f), 16.0f / 9.0f, 1.0f, 100.0f);
    const auto planes = frustumPlanes(projection);

    // The view looks down -z. A point just inside the near plane is in; just outside is out.
    REQUIRE_FALSE(pointOutside(planes, glm::vec3(0.0f, 0.0f, -1.5f)));
    REQUIRE(pointOutside(planes, glm::vec3(0.0f, 0.0f, -0.5f)));

    // And the far plane closes the volume, which is the half the four sides cannot do.
    REQUIRE_FALSE(pointOutside(planes, glm::vec3(0.0f, 0.0f, -99.0f)));
    REQUIRE(pointOutside(planes, glm::vec3(0.0f, 0.0f, -101.0f)));

    // Behind the eye is outside, which is the case a sides-only test gets wrong.
    REQUIRE(pointOutside(planes, glm::vec3(0.0f, 0.0f, 10.0f)));

    // Wide of the frustum at a depth that is otherwise fine.
    const auto halfHeight = std::tan(glm::radians(75.0f) * 0.5f) * 50.0f;
    const auto halfWidth = halfHeight * (16.0f / 9.0f);
    REQUIRE_FALSE(pointOutside(planes, glm::vec3(halfWidth * 0.9f, 0.0f, -50.0f)));
    REQUIRE(pointOutside(planes, glm::vec3(halfWidth * 1.2f, 0.0f, -50.0f)));
    REQUIRE_FALSE(pointOutside(planes, glm::vec3(0.0f, halfHeight * 0.9f, -50.0f)));
    REQUIRE(pointOutside(planes, glm::vec3(0.0f, halfHeight * 1.2f, -50.0f)));
}

// The normalisation is what lets a caller state its slack as a distance. Without it the plane
// offsets carry whatever scale the projection had, and an epsilon in world units means nothing.
TEST_CASE("frustum planes are normalised so the offset is a world distance", "[graphics][frustum]")
{
    const auto projection = glm::perspective(glm::radians(60.0f), 1.5f, 0.5f, 250.0f);
    const auto planes = frustumPlanes(projection);

    for (const auto& plane : planes)
    {
        REQUIRE(glm::length(glm::vec3(plane)) == Approx(1.0f).margin(1e-5f));
    }

    // The near plane is the fifth by construction, and a point 3 units in front of it should be
    // exactly 3 units on the positive side.
    const auto nearPlane = planes[4];
    const auto point = glm::vec3(0.0f, 0.0f, -3.5f);
    REQUIRE(glm::dot(glm::vec3(nearPlane), point) + nearPlane.w == Approx(3.0f).margin(1e-4f));
}

// The cascade case. An orthographic volume is the same extraction with no branch, and this is the
// test that says the box test agrees with the box.
TEST_CASE("frustum planes bound an orthographic volume", "[graphics][frustum]")
{
    const auto projection = glm::ortho(-10.0f, 10.0f, -5.0f, 5.0f, 0.0f, 40.0f);
    const auto planes = frustumPlanes(projection);

    REQUIRE_FALSE(pointOutside(planes, glm::vec3(0.0f, 0.0f, -20.0f)));
    REQUIRE(pointOutside(planes, glm::vec3(11.0f, 0.0f, -20.0f)));
    REQUIRE(pointOutside(planes, glm::vec3(0.0f, 6.0f, -20.0f)));

    // The depth range: an orthographic volume clips at both ends exactly as a perspective one does,
    // which is the fact the cull's exactness rests on for a shadow cascade.
    REQUIRE(pointOutside(planes, glm::vec3(0.0f, 0.0f, 1.0f)));
    REQUIRE(pointOutside(planes, glm::vec3(0.0f, 0.0f, -41.0f)));
}

TEST_CASE("a box straddling a plane is kept", "[graphics][frustum]")
{
    const auto projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 40.0f);
    const auto planes = frustumPlanes(projection);

    // Centre is outside the right-hand plane; the box reaches back across it, so it must be kept.
    REQUIRE_FALSE(boxOutside(planes, glm::vec3(12.0f, 0.0f, -20.0f), glm::vec3(3.0f)));

    // The same box moved clear of it is rejected.
    REQUIRE(boxOutside(planes, glm::vec3(20.0f, 0.0f, -20.0f), glm::vec3(3.0f)));
}

// Arvo's transform: a rotated box folds into a larger axis-aligned one, never a smaller one.
// Understating it here is what would cull something visible.
TEST_CASE("a rotated box is bounded conservatively", "[graphics][frustum]")
{
    const auto projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 40.0f);
    const auto planes = frustumPlanes(projection);

    const auto rotation = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    const auto placement = glm::translate(glm::mat4(1.0f), glm::vec3(11.2f, 0.0f, -20.0f)) * rotation;

    // A unit box rotated 45 degrees reaches sqrt(2) along x, so it still crosses the plane at 10
    // even though its unrotated half-extent of 1 would not have.
    REQUIRE_FALSE(boxOutside(planes, glm::vec3(0.0f), glm::vec3(1.0f), placement));

    // Far enough out that even the rotated reach cannot get back in.
    const auto clear = glm::translate(glm::mat4(1.0f), glm::vec3(14.0f, 0.0f, -20.0f)) * rotation;
    REQUIRE(boxOutside(planes, glm::vec3(0.0f), glm::vec3(1.0f), clear));
}

TEST_CASE("scale carries into the world bounds", "[graphics][frustum]")
{
    const auto projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 40.0f);
    const auto planes = frustumPlanes(projection);

    const auto placement =
        glm::translate(glm::mat4(1.0f), glm::vec3(14.0f, 0.0f, -20.0f)) * glm::scale(glm::mat4(1.0f), glm::vec3(10.0f));

    // Half-extent 1 scaled by 10 reaches from 4 to 24, so it crosses the plane at 10.
    REQUIRE_FALSE(boxOutside(planes, glm::vec3(0.0f), glm::vec3(1.0f), placement));

    // Unscaled, the same box sits clear of it.
    REQUIRE(boxOutside(planes, glm::vec3(14.0f, 0.0f, -20.0f), glm::vec3(1.0f)));
}

// The slack only ever keeps things, so a box on the knife edge draws. This is what makes the cull
// exact *with margin* rather than exact in exact arithmetic and approximate in floats.
TEST_CASE("the epsilon keeps a box on the boundary", "[graphics][frustum]")
{
    const auto projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 40.0f);
    const auto planes = frustumPlanes(projection);

    const auto justOutside = glm::vec3(10.0f + 0.005f, 0.0f, -20.0f);

    REQUIRE(aabbOutsideFrustum(planes, glm::mat4(1.0f), justOutside, glm::vec3(1e-6f), 0.0f));
    REQUIRE_FALSE(aabbOutsideFrustum(planes, glm::mat4(1.0f), justOutside, glm::vec3(1e-6f), 0.01f));
}

// The one that matters, and the one whose failure mode is a shadow quietly going missing.
//
// A cascade is fitted with its camera pulled back along the light by `casterExtent`, so a caster
// standing between the light and the slice is inside the volume and must be kept. Beyond that reach
// the rasteriser's own depth clipping already rejects it, so culling it is not a decision this test
// is making — it is agreeing with what the hardware does.
TEST_CASE("a cascade keeps casters up to its caster extent and no further", "[graphics][frustum]")
{
    constexpr auto casterExtent = 250.0f;
    constexpr auto resolution = 2048u;

    const auto lightDirection = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f));
    const auto fit = fitCascade(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), 75.0f, 16.0f / 9.0f, 1.0f, 40.0f,
                                lightDirection, resolution, casterExtent);

    const auto view = glm::lookAt(fit.position, fit.position + fit.direction, fit.roll);
    const auto projection =
        glm::ortho(fit.volume.left, fit.volume.right, fit.volume.bottom, fit.volume.top, fit.nearPlane, fit.farPlane);
    const auto planes = frustumPlanes(projection * view);

    // The centre of the slice: unambiguously in.
    const auto sliceCentre = fit.position + fit.direction * (fit.volume.right + casterExtent);
    REQUIRE_FALSE(pointOutside(planes, sliceCentre));

    // A caster most of the way up-light towards the near plane still casts into this cascade.
    REQUIRE_FALSE(pointOutside(planes, sliceCentre - lightDirection * (casterExtent * 0.8f)));

    // Past the volume the rasteriser has already clipped it, so the cull agrees rather than decides.
    REQUIRE(pointOutside(planes, sliceCentre - lightDirection * (casterExtent + fit.volume.right + 50.0f)));
}
