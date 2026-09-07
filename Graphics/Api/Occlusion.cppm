module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <glm/glm.hpp>

export module raceengine.graphics:Occlusion;

import raceengine.graphics.models;

namespace raceengine
{

// The other half of :Frustum. That one answers "can this view see the box at all"; this one answers
// "is the box behind something the view has already drawn". Both are pure functions of their
// arguments, for the same reason: this is code where a sign error still renders a plausible picture
// with something missing from the middle of it, and it has to be pinnable without a device.
//
// **This test is not exact, and the difference from the frustum test is the whole thing to
// understand about it.** `aabbOutsideFrustum` rejects geometry the rasteriser would have clipped
// anyway, so skipping the draw records the same image. This one rejects geometry that was hidden in
// a frame that has *already been submitted*, and asserts it is still hidden now. That assertion is
// true for a still camera and false, briefly, for one that has just swung round a corner. What
// bounds the error is stated below and knobbed on the camera; what it costs when the bound is too
// tight is a primitive that pops in a frame or two late, and what it costs when the bound is too
// loose is nothing but draws.
//
// Three things keep the *within-frame* half of the test conservative, so that everything left over
// really is staleness and not arithmetic:
//
//   - a cell holds the farthest distance anywhere inside it, and zero where any pixel of it was
//     open (OcclusionGrid says why), so a cell can only ever under-claim what it covers;
//   - the box is the one the POSITION accessor declared, carried into world space by Arvo's method
//     exactly as the frustum test carries it, which grows it rather than shrinking it;
//   - a box that is not wholly on screen is refused outright. The off-screen part of it writes no
//     fragments *this* frame, so testing only the visible cells would be sound for a still camera —
//     but a camera that rotates brings that part on screen, and the grid never measured it. This is
//     the one place where the temporal half of the test dictates the spatial half.

// Whether a primitive's bounding box, carried into world space by `localToWorld`, was wholly behind
// the geometry this grid recorded.
//
// `dilation` grows the box in every direction, in world units, and is how camera *translation*
// since the grid was captured is paid for: a box that was hidden while standing this much bigger is
// one an eye that has since moved by that much is unlikely to have found a way past. The caller
// measures it as the distance from `grid.eye` to where the eye stands now.
//
// `cellMargin` widens the tested rectangle by that many cells on each side. It is not paying for
// camera movement — the two above do that — but for the seam between two mappings that are allowed
// to disagree by a cell: this projects a rectangle into clip space and floors it, while the shader
// that filled the grid divided the source's pixels into cells by integer division. Widening can only
// ever add cells that have to be occluded too, so it is conservative by construction. Zero makes the
// test as sharp as the grid can state.
//
// **Rotation needs no slack at all, and that is worth stating because it looks as though it should.**
// Whether a box is hidden is a property of where the eye *is*, not of where it is pointing, so a
// camera that has only turned since the grid was photographed has not disoccluded anything. What
// turning does is bring geometry in from the edge of the frame — which is exactly the case the
// wholly-on-screen refusal above already rejects.
export [[nodiscard]] inline bool aabbOccluded(const OcclusionGrid& grid, const glm::mat4& localToWorld,
                                              const glm::vec3& localCentre, const glm::vec3& localHalfExtent,
                                              const float dilation = 0.0f, const std::uint32_t cellMargin = 1u)
{
    if (grid.width == 0 || grid.height == 0 ||
        grid.farthest.size() != static_cast<std::size_t>(grid.width) * static_cast<std::size_t>(grid.height))
    {
        return false;
    }

    // Arvo, the same as :Frustum — the centre through the matrix as a point and the half-extent
    // scaled by the absolute upper 3x3 — and then grown by the staleness slack.
    const auto worldCentre = glm::vec3(localToWorld * glm::vec4(localCentre, 1.0f));

    const auto basis = glm::mat3(localToWorld);
    const auto absoluteBasis = glm::mat3(glm::abs(basis[0]), glm::abs(basis[1]), glm::abs(basis[2]));
    const auto worldHalfExtent = absoluteBasis * localHalfExtent + glm::vec3(std::max(dilation, 0.0f));

    auto minimumX = 1.0f;
    auto maximumX = -1.0f;
    auto minimumY = 1.0f;
    auto maximumY = -1.0f;
    auto nearest = std::numeric_limits<float>::max();

    for (auto corner = 0; corner < 8; corner++)
    {
        const auto world =
            worldCentre + glm::vec3((corner & 1) != 0 ? worldHalfExtent.x : -worldHalfExtent.x,
                                    (corner & 2) != 0 ? worldHalfExtent.y : -worldHalfExtent.y,
                                    (corner & 4) != 0 ? worldHalfExtent.z : -worldHalfExtent.z);

        // The distance in front of the eye, which is the quantity the cells hold: the view looks
        // down negative z, so it is the negated one, and PrepassFragmentShader states the same sign
        // on the other side of the divide.
        const auto viewDepth = -(grid.view * glm::vec4(world, 1.0f)).z;

        const auto clip = grid.viewProjection * glm::vec4(world, 1.0f);

        // A box with a corner at or behind the eye has no projection worth the name — the divide
        // flips its sign and the rectangle comes out inside out — and a box the eye is inside is
        // exactly the case that must never be culled. Both are this one test.
        if (clip.w <= 0.0f || viewDepth <= grid.nearPlane)
        {
            return false;
        }

        const auto ndc = glm::vec2(clip) / clip.w;

        minimumX = std::min(minimumX, ndc.x);
        maximumX = std::max(maximumX, ndc.x);
        minimumY = std::min(minimumY, ndc.y);
        maximumY = std::max(maximumY, ndc.y);
        nearest = std::min(nearest, viewDepth);
    }

    // Not wholly on screen: refused, for the reason at the head of this file. The frustum test owns
    // the wholly-off-screen case and has already run.
    if (minimumX < -1.0f || maximumX > 1.0f || minimumY < -1.0f || maximumY > 1.0f)
    {
        return false;
    }

    const auto width = static_cast<float>(grid.width);
    const auto height = static_cast<float>(grid.height);

    // Clip x runs left to right and the viewport does not flip it, so a column is the obvious map.
    // Clip y is the opposite: the scene pass renders through a **negative** viewport height, so
    // clip y = +1 lands on framebuffer row 0 and the fullscreen reduction inherits that one to one.
    // Getting this upside down is a cull that works perfectly while looking at the horizon.
    const auto firstColumn = static_cast<int>(std::floor((minimumX * 0.5f + 0.5f) * width));
    const auto lastColumn = static_cast<int>(std::floor((maximumX * 0.5f + 0.5f) * width));
    const auto firstRow = static_cast<int>(std::floor((0.5f - maximumY * 0.5f) * height));
    const auto lastRow = static_cast<int>(std::floor((0.5f - minimumY * 0.5f) * height));

    const auto margin = static_cast<int>(cellMargin);
    const auto columnBegin = std::max(firstColumn - margin, 0);
    const auto columnEnd = std::min(lastColumn + margin, static_cast<int>(grid.width) - 1);
    const auto rowBegin = std::max(firstRow - margin, 0);
    const auto rowEnd = std::min(lastRow + margin, static_cast<int>(grid.height) - 1);

    if (columnBegin > columnEnd || rowBegin > rowEnd)
    {
        return false;
    }

    for (auto row = rowBegin; row <= rowEnd; row++)
    {
        const auto rowBase = static_cast<std::size_t>(row) * static_cast<std::size_t>(grid.width);

        for (auto column = columnBegin; column <= columnEnd; column++)
        {
            const auto cell = grid.farthest[rowBase + static_cast<std::size_t>(column)];

            // Open, so nothing behind it is proven hidden; or not far enough forward to cover the
            // nearest corner of the box. Either way this primitive gets drawn.
            if (cell <= 0.0f || cell >= nearest)
            {
                return false;
            }
        }
    }

    return true;
}

} // namespace raceengine
