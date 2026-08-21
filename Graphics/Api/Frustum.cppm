module;

#include <array>

#include <glm/glm.hpp>

export module raceengine.graphics:Frustum;

namespace raceengine
{

// The arithmetic a visibility test is made of, and nothing else: no storage, no device, no service.
// Both functions are pure functions of their arguments, which is what lets the plane extraction and
// the box test be pinned without a GPU — and this is exactly the kind of code where a sign error
// still renders a plausible picture, just with something missing from a corner of it.
//
// The whole point of this file is that the test is *exact*, not merely conservative-looking. A
// primitive it rejects writes literally zero fragments, so a pass that skips the draw records the
// same image as a pass that made it. That claim rests on three facts about this engine:
//
//   - `CameraService::projectionMatrix` keeps glm's GL depth convention (z in -w..w) for *both* the
//     perspective and the orthographic arm, and the Vulkan 0..1 correction is a post-multiply
//     applied later, when the backend fills DrawData. So the textbook Gribb-Hartmann extraction
//     below is right for every camera this engine has, with no projection-type branch and no depth
//     convention special case.
//   - The negative viewport height the scene pass uses does not enter into it. The viewport
//     transform runs *after* clipping; clip-space rejection is -w <= x,y <= w whichever way the
//     framebuffer's y points. A y flip here would be a bug that deleted half the picture.
//   - `depthClampEnable` is VK_FALSE everywhere (it is never set, and the rasterization state is
//     value-initialised), so depth clipping is live and the near and far planes reject as surely as
//     the four sides do. That is what makes the cascade case exact: a caster further up-light than
//     the cascade's own box is *already* clipped today, so testing against the box is testing
//     against what the rasteriser would have done anyway.
//
// The last one is worth stating twice, because the usual advice points the other way. A cascade is
// fitted with the camera pulled back along the light by `casterExtent` (see :ShadowCascades), which
// is what catches a building standing between the light and the slice. That volume *is* the set of
// geometry that can write a texel of this cascade. Extending it further toward the light before
// testing would only weaken the cull; shrinking it — testing against the slice sphere rather than
// the fitted box — is what silently deletes shadows.

// The six planes of a clip volume, in world space, as ax + by + cz + d = 0 with the normal pointing
// *inward*: a point is inside the volume when it is on the positive side of all six.
//
// Gribb-Hartmann: a clip-space coordinate is inside when -w <= x <= w (and likewise y and z), and
// each of those six inequalities is a row of the matrix added to or subtracted from the w row. The
// result is a plane in whatever space the matrix started in, which here is world space because the
// argument is a view-projection.
//
// Each plane is normalised, so `w` is a signed distance in world units rather than in whatever
// scale the matrix happened to carry. That costs six square roots per view and is what lets the
// caller state its epsilon as a distance instead of as an uninterpretable number.
export [[nodiscard]] inline std::array<glm::vec4, 6> frustumPlanes(const glm::mat4& viewProjection)
{
    // glm is column-major, so `viewProjection[column][row]`: this is the transpose of the way the
    // rows are written in the literature, and reading it the other way round produces a frustum
    // that is plausibly shaped and points somewhere else entirely.
    const auto row = [&viewProjection](const int index)
    {
        return glm::vec4(viewProjection[0][index], viewProjection[1][index], viewProjection[2][index],
                         viewProjection[3][index]);
    };

    const auto rowX = row(0);
    const auto rowY = row(1);
    const auto rowZ = row(2);
    const auto rowW = row(3);

    std::array<glm::vec4, 6> planes = {rowW + rowX, rowW - rowX, rowW + rowY, rowW - rowY, rowW + rowZ, rowW - rowZ};

    for (auto& plane : planes)
    {
        // The length of the normal, not of the whole plane: `w` is the offset and scales with the
        // normal, which is the entire reason this normalisation makes `w` a distance.
        const auto length = glm::length(glm::vec3(plane));
        if (length > 0.0f)
        {
            plane /= length;
        }
    }

    return planes;
}

// Whether a primitive's bounding box, carried into world space by `localToWorld`, lies wholly
// outside the volume those planes bound.
//
// The box arrives as the centre and half-extent the glTF POSITION accessor declared, in the mesh's
// own local space, and is transformed by Arvo's method: the centre goes through the matrix as a
// point, and the half-extent is scaled by the absolute value of the upper 3x3. That is correct for
// any affine transform — rotation folds the box into a larger axis-aligned one rather than a
// sheared one — and it is conservative in the only direction that is safe, since the transformed
// box always contains the transformed geometry.
//
// The test itself is the positive-vertex ("p-vertex") one: for each plane, the corner of the box
// furthest along the plane's inward normal is the last one that could still be inside, so if *that*
// corner is behind the plane the whole box is. Written with the dot product of the absolute normal
// against the half-extent, which is that corner's offset from the centre without a branch.
//
// `epsilon` is a slack in world units, and it exists because the extraction and this dot product
// both round. Rejecting only when the box is behind by *more* than epsilon turns "exact in exact
// arithmetic" into "exact with margin": a primitive that rounding places on the knife edge is kept
// and drawn, which costs a draw and can never cost a pixel.
export [[nodiscard]] inline bool aabbOutsideFrustum(const std::array<glm::vec4, 6>& planes,
                                                    const glm::mat4& localToWorld, const glm::vec3& localCentre,
                                                    const glm::vec3& localHalfExtent, const float epsilon = 0.01f)
{
    const auto worldCentre = glm::vec3(localToWorld * glm::vec4(localCentre, 1.0f));

    const auto basis = glm::mat3(localToWorld);
    const auto absoluteBasis = glm::mat3(glm::abs(basis[0]), glm::abs(basis[1]), glm::abs(basis[2]));
    const auto worldHalfExtent = absoluteBasis * localHalfExtent;

    for (const auto& plane : planes)
    {
        const auto normal = glm::vec3(plane);

        // How far the furthest-inward corner sits from the centre, along this plane's normal.
        const auto reach = glm::dot(glm::abs(normal), worldHalfExtent);

        if (glm::dot(normal, worldCentre) + plane.w + reach < -epsilon)
        {
            return true;
        }
    }

    return false;
}

} // namespace raceengine
