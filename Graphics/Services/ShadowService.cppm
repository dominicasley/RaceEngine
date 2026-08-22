module;

#include <expected>
#include <string>
#include <utility>

#include <glm/glm.hpp>

export module raceengine.graphics:ShadowService;

import :CameraService;
import :RenderContract;
import :ShadowCascades;
import raceengine.graphics.models;

namespace raceengine
{

// The scene's cascaded shadow map, as cameras.
//
// Every camera already renders into its own framebuffer and `Engine::step` records one view per
// camera, so a cascade needs no frame concept of its own: it is one more depth-only orthographic
// camera, refitted to a slice of the view frustum once per frame. This service is the two things
// that are not already there — creating that set of cameras, and moving them.
export class ShadowService
{
private:
    CameraService& cameraService;

public:
    explicit ShadowService(CameraService& cameraService);

    // Gives a scene a cascaded shadow map for one directional light. The cameras are appended to
    // the scene's own deque, so they are torn down with it and nothing else has to remember them.
    // A scene that already had cascades is not given a second set.
    [[nodiscard]] std::expected<void, std::string> enable(Scene& scene, Light& light, Camera& viewCamera,
                                                          const CreateShadowCascadesDTO& createShadowCascadesDTO) const;

    // Refits every cascade to the view camera's current frustum. Called once per frame, before the
    // cameras' matrices are recomputed — the fit *is* what those matrices are computed from.
    void update(Scene& scene) const;
};

} // namespace raceengine
