module;

#include <expected>
#include <functional>
#include <string>
#include <utility>

export module raceengine.graphics:SceneService;

import :RenderableEntityService;
import :CameraService;
import :SceneManagerService;
import raceengine.graphics.models;

namespace raceengine
{

export class SceneService
{
private:
    RenderableEntityService& renderableEntityService;
    CameraService& cameraService;
    SceneManagerService& sceneManagerService;

public:
    explicit SceneService(RenderableEntityService& renderableEntityService, CameraService& cameraService,
                          SceneManagerService& sceneManagerService);
    void update(Scene& scene, float delta) const;
    [[nodiscard]] RenderableModel& createEntity(Scene& scene, const CreateRenderableModelDTO& entityDescriptor) const;
    // The camera is owned by the scene's deque, so what a successful call hands back is a
    // reference into it; reference_wrapper is what lets that ride in an expected.
    [[nodiscard]] std::expected<std::reference_wrapper<Camera>, std::string> createCamera(Scene& scene) const;
    [[nodiscard]] Light& createLight(Scene& scene) const;
    [[nodiscard]] RenderableModel& getModel(Scene& scene, unsigned int index) const;
    [[nodiscard]] Camera& getCamera(Scene& scene, unsigned int index) const;
};

// The scene's share of a simulation tick: settle the scene graph once every writer for this
// tick has run. Resolving here rather than lazily from the draw loop means world transforms
// are computed once per tick instead of once per node per camera, and it means the next tick's
// behaviours read positions that already include the previous tick's writes — including a
// parent's, which a child cannot see until the parent's version has been recomputed.
// modelMatrix is idempotent, so a node nothing touched costs a flag test.
//
// delta is unnamed because the scene owns no integrator of its own yet: the one time-varying
// piece of scene state, animation playback, is advanced inside the renderers' draw path
// (recorded in docs/architecture-review.md as a defect — it advances once per camera per
// frame). Moving it here is the fix, and it belongs with that change, not this one.

} // namespace raceengine
