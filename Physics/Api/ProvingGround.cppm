module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:ProvingGround;

namespace raceengine
{

// The surface the vehicle is validated against, generated rather than authored.
//
// A flat plane would let every one of the contact-patch failures pass unnoticed — aggregation
// across a tilted patch, a migrating patch centre, blended grip at a boundary — because none of
// them has anything to act on. So the ground carries a kerb, a surface boundary, a camber change
// and a ramp from the first day, and it is *generated* so that it is deterministic, diffable, and
// does not depend on an asset that can be lost. (One already was: see the note on plane.glb.)
//
// SI, like everything else under Physics: metres. +x is lateral, +z is along the ground, +y is up,
// matching the engine's handedness.

// A coarse label and deliberately not the friction. A real circuit states nine surfaces where this
// has five kinds, and the difference between a road, a pit lane, a painted line and a run-off is
// entirely in the numbers `SurfaceMaterial` carries — so quantising them onto this enum would throw
// a 0.98 road and a 0.95 pit lane onto the same value for nothing. What is worth a *kind* is what
// something other than the tire will one day branch on: dust off a gravel trap, a wall that is not
// a surface a car drives on at all.
export enum class SurfaceKind : std::uint32_t { Tarmac, Kerb, Grass, Gravel, Wall };

// What a surface is worth to a tire, separately from where it is. Kept beside the mesh rather than
// baked into it because the tire model scales its own mu by this at runtime — the seam the deferred
// thermal model needs, and the reason `gripMultiplier` is not folded into a Pacejka coefficient.
export struct SurfaceMaterial
{
    double gripMultiplier = 1.0;
    // Amplitude, in metres, of the fine surface roughness a tire feels but the mesh does not
    // resolve. Nothing consumes it this milestone; it is a channel the sampler already carries.
    double bumpiness = 0.0;
    // Velocity-proportional drag a surface adds beyond what the tire model already produces —
    // ploughing through gravel rather than rolling over it. Assetto Corsa's `surfaces.ini` states
    // one per surface and a real circuit's export carries it, so it is read and kept here rather
    // than discarded at the boundary. Nothing consumes it this milestone either.
    double damping = 0.0;
    SurfaceKind kind = SurfaceKind::Tarmac;
};

export enum class FeatureKind : std::uint32_t {
    // A raised kerb parallel to the direction of travel, so a wheel can be run onto it at an angle
    // and half-mount it. This is the case single-ray contact gets visibly wrong.
    Kerb,
    // Tarmac one side, grass the other, with the small lip a real one has. A wheel straddling it
    // must come out with grip between the two rather than snapping from one to the other.
    SurfaceBoundary,
    // The ground banks, so the contact plane is not the world's horizontal and the suspension has
    // to be solved against a surface that is not flat under it.
    Camber,
    // Up and then nothing: the launch that criterion 11 asks for a clean arc off.
    Ramp,
    // A constant grade along the direction of travel. What criteria 2 and 8 need: somewhere to park
    // a car on the brakes and find out whether it stays there, and somewhere to creep up and down
    // at walking pace and find out whether the tire model is still finite when it does.
    Slope,
    // A wall down one side. The only feature that is not a height — a heightfield cannot express a
    // vertical face at all — so it is emitted as its own geometry after the grid, and it is the one
    // thing criterion 2's first half needs: something for a car to rest against.
    Barrier
};

export struct Feature
{
    FeatureKind kind = FeatureKind::Kerb;
    // The band this feature occupies along z, in metres. Bands are evaluated in order and a later
    // one wins, so overlapping them is a way of composing rather than an error.
    double from = 0.0;
    double to = 0.0;
};

export struct ProvingGroundDescriptor
{
    double length = 200.0;
    double width = 40.0;
    // The grid pitch. A tire's contact patch is about 0.2 m across, and the sample grid inside it
    // is finer still, so anything much coarser than this would have the patch spanning one triangle
    // and aggregating nothing.
    double cellSize = 0.25;

    // Height of the kerb's flat top, and the width of the chamfer up onto it.
    double kerbHeight = 0.05;
    double kerbChamfer = 0.30;
    // Where the kerb's inner edge sits, measured from the centreline towards +x.
    double kerbInnerEdge = 3.0;

    // How far the grass sits below the tarmac at the boundary.
    double boundaryLip = 0.02;
    // Peak bank angle of the camber section, radians. Six degrees.
    double camberAngle = 0.10472;
    double rampHeight = 1.20;
    // Fifteen degrees, which is what criterion 2 asks a parked car to hold on.
    double slopeAngle = 0.26180;

    // Where the barrier stands and how tall it is. Positive x, so a car pushed to its right finds it.
    double barrierX = 4.0;
    double barrierHeight = 0.9;

    // How far a feature takes to come up to full height at each end of its band, in metres.
    //
    // Without it a feature switches on abruptly at its own boundary, and a kerb that begins as a
    // fifty-millimetre vertical wall across the road is not a kerb — it is a step, and a wheel
    // hitting it produces a load spike that looks exactly like a contact bug. Real kerbs are ramped
    // at the ends and so is this one.
    double featureEasing = 1.0;

    std::vector<Feature> features;
};

export struct SurfaceSample
{
    double height = 0.0;
    SurfaceKind kind = SurfaceKind::Tarmac;
};

export struct SurfaceMesh
{
    std::vector<glm::dvec3> vertices;
    // Three per triangle, counter-clockwise seen from above, which is the winding the engine's
    // front face already is.
    std::vector<std::uint32_t> indices;
    // One per triangle, indexing `materials`. Per-triangle rather than per-vertex because a
    // boundary between two surfaces is a line the mesh is cut along, not a blend across a face —
    // the blending belongs to the sampler that aggregates a contact patch, where it can be weighted
    // by how much of the patch is on each side.
    std::vector<std::uint32_t> surfaces;
    std::vector<SurfaceMaterial> materials;

    [[nodiscard]] std::size_t triangleCount() const
    {
        return indices.size() / 3;
    }
};

// The materials the generator emits, one per kind it can emit and in `SurfaceKind` order, so those
// kinds index their own material. It stops at Grass because that is where the generator stops:
// gravel and walls are things a *circuit* carries, and the table an authored track brings with it is
// its own — see `PhysicsWorld::materials`.
export [[nodiscard]] std::vector<SurfaceMaterial> defaultSurfaceMaterials();

// The layout the acceptance criteria need: a long flat run to settle, accelerate, brake and hold a
// skidpad on, then each feature in its own band with flat ground either side so a test can drive
// onto exactly one of them.
export [[nodiscard]] ProvingGroundDescriptor defaultProvingGround();

// Height and surface at a point, which is the whole of the ground's definition: the mesh below is
// only this function sampled on a grid. Pure and total — every (x, z) has an answer, including
// outside the ground's extent, so a query that walks off the edge reads flat tarmac rather than
// nothing.
export [[nodiscard]] SurfaceSample sampleProvingGround(const ProvingGroundDescriptor& descriptor, const double x,
                                                       const double z);

// The generated mesh. Reported rather than returned bare: a descriptor with no extent, or a cell
// size that would make one, is a configuration error and not something to hand back an empty mesh
// for.
export [[nodiscard]] std::expected<SurfaceMesh, std::string>
generateProvingGround(const ProvingGroundDescriptor& descriptor);

} // namespace raceengine
