module;

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.graphics.models:OcclusionGrid;

namespace raceengine
{

// What a frame's own geometry says about what the *next* frame need not draw.
//
// A coarse grid over the view, one cell per few pixels, each cell holding the **farthest** distance
// in front of the eye that the occlusion prepass wrote anywhere inside it — or zero where any pixel
// of that cell saw no geometry at all. Both halves of that sentence are what makes the grid usable
// as an occluder rather than merely as a picture of depth:
//
//   - **farthest, not nearest.** A cell whose farthest surface is 40 units away is a cell where
//     every pixel is covered by something within 40 units. Anything behind 40 units in that cell is
//     hidden by geometry the frame has already rasterised. Taking the nearest instead would claim
//     occlusion for a cell that is mostly open sky with one lamp post through it.
//   - **zero means open.** The prepass clears its distance channel to zero and writes a distance
//     only where something drew (PrepassFragmentShader states that convention), so a cell holding
//     zero is a cell with sky — or with nothing yet drawn — somewhere inside it, and a cell with
//     sky in it can never prove anything behind it invisible.
//
// The grid is a **photograph of a frame that has already been submitted**. It carries the view it
// was rasterised with rather than being read against the camera's current one, because the copy
// reaches the CPU a fixed number of submissions after the frame that recorded it — the same
// deferred readback the exposure meter and the light probes use, and a lag in submissions rather
// than in wall time, so it survives the frame gate. Every consumer therefore projects into *these*
// matrices and not into the camera's, and the camera's movement since is handled as a stated slack
// rather than as an approximation nobody wrote down (see aabbOccluded).
export struct OcclusionGrid
{
    // Cells across and down. Row 0 is the **top** of the screen: the scene pass renders through a
    // negative viewport height, so its attachment's first row is clip y = +1, and the fullscreen
    // pass that reduces it maps one to one onto that. aabbOccluded is the only thing that has to
    // know, and it says so where it does the mapping.
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // width * height entries, row major. Distances in front of the eye, in world units; zero where
    // the cell is open.
    std::vector<float> farthest;
    // The view this was rasterised with. Both, because the two answer different questions: the
    // view-projection gives the cell a box lands in, and the view alone gives the distance in front
    // of the eye that the cells are measured in — which is what the prepass wrote and is neither of
    // the depth buffer's own conventions.
    glm::mat4 viewProjection{1.0f};
    glm::mat4 view{1.0f};
    // Where the eye stood, and where its near plane was. The first is what the staleness slack is
    // measured from; the second is what says a box straddling the eye cannot be projected at all.
    glm::vec3 eye{0.0f};
    float nearPlane = 0.0f;
};

} // namespace raceengine
