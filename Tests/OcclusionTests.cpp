#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

import raceengine;

using raceengine::aabbOccluded;
using raceengine::OcclusionGrid;

namespace
{

constexpr auto gridWidth = std::uint32_t{16};
constexpr auto gridHeight = std::uint32_t{16};
constexpr auto nearPlane = 1.0f;
constexpr auto farPlane = 1000.0f;

// An eye at the origin looking down negative z, which is the frame everything in the renderer works
// in and the one PrepassFragmentShader negates z against. Every cell is filled with one distance, so
// a test states its occluder as a single number and reads the result as "behind that or not".
[[nodiscard]] OcclusionGrid gridAt(const float occluderDistance)
{
    OcclusionGrid grid;
    grid.width = gridWidth;
    grid.height = gridHeight;
    grid.farthest.assign(static_cast<std::size_t>(gridWidth) * gridHeight, occluderDistance);
    grid.view = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    grid.viewProjection = glm::perspective(glm::radians(90.0f), 1.0f, nearPlane, farPlane) * grid.view;
    grid.eye = glm::vec3(0.0f);
    grid.nearPlane = nearPlane;
    return grid;
}

[[nodiscard]] bool occluded(const OcclusionGrid& grid, const glm::vec3& centre, const glm::vec3& halfExtent,
                            const float dilation = 0.0f, const std::uint32_t margin = 0u)
{
    return aabbOccluded(grid, glm::mat4(1.0f), centre, halfExtent, dilation, margin);
}

} // namespace

TEST_CASE("a box behind the recorded geometry is occluded", "[graphics][occlusion]")
{
    const auto grid = gridAt(10.0f);

    // Twenty units in front of the eye, behind an occluder that covers every cell at ten.
    REQUIRE(occluded(grid, glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(1.0f)));
}

TEST_CASE("a box in front of the recorded geometry is drawn", "[graphics][occlusion]")
{
    const auto grid = gridAt(10.0f);

    REQUIRE_FALSE(occluded(grid, glm::vec3(0.0f, 0.0f, -5.0f), glm::vec3(1.0f)));
}

// The nearest corner decides, not the centre: a box straddling the occluder has something in front
// of it and must be drawn whole.
TEST_CASE("a box straddling the occluder is drawn", "[graphics][occlusion]")
{
    const auto grid = gridAt(10.0f);

    REQUIRE_FALSE(occluded(grid, glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(3.0f)));
}

// Zero is the prepass's own "nothing drew here", and one such cell anywhere under the box is enough
// to keep it. This is the rule that stops a silhouette against the sky being culled.
TEST_CASE("an open cell keeps the box", "[graphics][occlusion]")
{
    auto grid = gridAt(10.0f);
    grid.farthest[0] = 0.0f;

    // Filling the frame, so its rectangle reaches the corner the hole is in.
    REQUIRE_FALSE(occluded(grid, glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(15.0f)));

    // Small and in the middle of the frame, which never touches that corner.
    REQUIRE(occluded(grid, glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(0.5f)));
}

// **The test that catches an upside-down grid.** The scene pass renders through a negative viewport
// height, so clip y = +1 is framebuffer row 0 and the reduction inherits that: a box *above* the
// centre of the screen has to read the *first* rows. Flipped, this passes every test that fills the
// whole grid and deletes the wrong half of every real frame.
TEST_CASE("clip y maps to the grid's rows top down", "[graphics][occlusion]")
{
    auto grid = gridAt(10.0f);

    // Solid across the top half, open across the bottom half.
    for (auto row = gridHeight / 2; row < gridHeight; row++)
    {
        for (auto column = std::uint32_t{0}; column < gridWidth; column++)
        {
            grid.farthest[static_cast<std::size_t>(row) * gridWidth + column] = 0.0f;
        }
    }

    // A 90 degree square frustum, so at twenty units in front the frame is forty units tall: ten
    // above the axis is comfortably inside the top half and clear of the seam.
    REQUIRE(occluded(grid, glm::vec3(0.0f, 10.0f, -20.0f), glm::vec3(0.5f)));
    REQUIRE_FALSE(occluded(grid, glm::vec3(0.0f, -10.0f, -20.0f), glm::vec3(0.5f)));
}

// A box the eye is inside, or one with a corner behind it, has no projection worth the name. It is
// also exactly the case a wrong answer is most visible in.
TEST_CASE("a box around the eye is drawn", "[graphics][occlusion]")
{
    const auto grid = gridAt(10.0f);

    REQUIRE_FALSE(occluded(grid, glm::vec3(0.0f), glm::vec3(5.0f)));
    REQUIRE_FALSE(occluded(grid, glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(30.0f)));
}

// Not wholly on screen means refused: the off-screen part writes nothing this frame, but the grid
// never measured it and a camera that rotates brings it in.
TEST_CASE("a box crossing the screen edge is drawn", "[graphics][occlusion]")
{
    const auto grid = gridAt(10.0f);

    // At twenty units the frame is forty wide, so a box centred eighteen out with a five unit
    // half-extent hangs over the right edge.
    REQUIRE_FALSE(occluded(grid, glm::vec3(18.0f, 0.0f, -20.0f), glm::vec3(5.0f)));
    REQUIRE(occluded(grid, glm::vec3(10.0f, 0.0f, -20.0f), glm::vec3(5.0f)));
}

// The staleness slack only ever keeps geometry: a box grown until it reaches an open cell is drawn.
TEST_CASE("the dilation only ever keeps geometry", "[graphics][occlusion]")
{
    auto grid = gridAt(10.0f);
    grid.farthest[0] = 0.0f;

    const auto centre = glm::vec3(0.0f, 0.0f, -20.0f);

    REQUIRE(occluded(grid, centre, glm::vec3(0.5f)));
    REQUIRE_FALSE(occluded(grid, centre, glm::vec3(0.5f), 30.0f));
}

// The cell margin does the same thing at the grid's own resolution.
TEST_CASE("the cell margin only ever keeps geometry", "[graphics][occlusion]")
{
    auto grid = gridAt(10.0f);

    // A single open cell just off the middle of the grid.
    grid.farthest[static_cast<std::size_t>(gridHeight / 2) * gridWidth + gridWidth / 2 + 2] = 0.0f;

    const auto centre = glm::vec3(0.0f, 0.0f, -20.0f);

    REQUIRE(occluded(grid, centre, glm::vec3(0.5f), 0.0f, 0u));
    REQUIRE_FALSE(occluded(grid, centre, glm::vec3(0.5f), 0.0f, 4u));
}

// A grid nothing has been read back into yet must cull nothing at all, which is the first frames of
// every run.
TEST_CASE("an empty grid culls nothing", "[graphics][occlusion]")
{
    const OcclusionGrid empty;

    REQUIRE_FALSE(occluded(empty, glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(1.0f)));

    auto ragged = gridAt(10.0f);
    ragged.farthest.pop_back();

    REQUIRE_FALSE(occluded(ragged, glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(1.0f)));
}

// The box arrives in some mesh's local space, and the matrix is what puts it in the world. Arvo's
// method is shared with the frustum test, and the thing that matters here is that it is applied at
// all: an unrotated read of a rotated box is a box in the wrong place.
TEST_CASE("the box is carried into world space before it is tested", "[graphics][occlusion]")
{
    const auto grid = gridAt(10.0f);

    // A unit box at the local origin, moved twenty units in front of the eye by the matrix alone.
    const auto toWorld = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -20.0f));

    REQUIRE(aabbOccluded(grid, toWorld, glm::vec3(0.0f), glm::vec3(1.0f), 0.0f, 0u));

    // And five in front of it, where nothing is in the way.
    const auto toNear = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f));

    REQUIRE_FALSE(aabbOccluded(grid, toNear, glm::vec3(0.0f), glm::vec3(1.0f), 0.0f, 0u));
}
