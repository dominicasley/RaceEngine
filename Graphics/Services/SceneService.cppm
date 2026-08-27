module;

#include <expected>
#include <functional>
#include <string>
#include <utility>

#include <glm/glm.hpp>

export module raceengine.graphics:SceneService;

import :RenderableEntityService;
import :CameraService;
import :SceneManagerService;
import :ShaderService;
import raceengine.graphics.models;

namespace raceengine
{

export class SceneService
{
private:
    RenderableEntityService& renderableEntityService;
    CameraService& cameraService;
    SceneManagerService& sceneManagerService;
    // Held here rather than by RenderableEntityService because this service is constructed after the
    // renderer and that one before it; see RenderableEntityService::createModel.
    ShaderService& shaderService;

public:
    explicit SceneService(RenderableEntityService& renderableEntityService, CameraService& cameraService,
                          SceneManagerService& sceneManagerService, ShaderService& shaderService);
    void update(Scene& scene, float delta) const;
    [[nodiscard]] RenderableModel& createEntity(Scene& scene, const CreateRenderableModelDTO& entityDescriptor) const;
    // The camera is owned by the scene's deque, so what a successful call hands back is a
    // reference into it; reference_wrapper is what lets that ride in an expected.
    [[nodiscard]] std::expected<std::reference_wrapper<Camera>, std::string> createCamera(Scene& scene) const;
    // The DTO form, appended to the same deque: a layered frame's extra cameras are scene cameras
    // like the first one, recorded in the order they were appended, and that order is the frame's.
    [[nodiscard]] std::expected<std::reference_wrapper<Camera>, std::string>
    createCamera(Scene& scene, const CreateCameraDTO& createCameraDTO) const;
    [[nodiscard]] Light& createLight(Scene& scene) const;
    [[nodiscard]] RenderableModel& getModel(Scene& scene, unsigned int index) const;
    [[nodiscard]] Camera& getCamera(Scene& scene, unsigned int index) const;
    // The air every camera in this scene looks through, and the one thing about the scene's lighting
    // that is stated rather than photographed. It goes through the service rather than being written
    // onto the scene because the numbers have a domain — a negative scale height or an asymmetry past
    // one is not a thinner fog, it is a shader dividing by something it should not — and the domain is
    // worth one sentence at the point of the mistake rather than a NaN three passes downstream.
    //
    // There is no buffer to build and nothing to tear down, which is why this is a setter here and
    // not a service of its own: fog is nine numbers, and the two halves of the arithmetic that reads
    // them live in Graphics/Api/VolumetricFog.cppm and in the scene shaders.
    [[nodiscard]] std::expected<void, std::string> setFog(Scene& scene, const Fog& fog) const;
    [[nodiscard]] std::expected<void, std::string> setRain(Scene& scene, float rain) const;
    // The clouds over the scene, a per-scene statement exactly as the rain is. Zero coverage is the
    // clear sky the renderer has always drawn; type is the stratus-to-cumulus blend and means
    // nothing without coverage. See Scene::clouds.
    [[nodiscard]] std::expected<void, std::string> setClouds(Scene& scene, float coverage, float type) const;
    // Which attachment carries the marched cloud dome map. Stated by whoever built the world
    // camera's chain, because which pass's output is the cloud map is a fact about that chain; the
    // backend binds it to every shading view and to every probe face. See Scene::cloudMap.
    void setCloudMap(Scene& scene, const Resource<FboAttachment>& cloudMap) const;
    // Where the rain's glass is going: the ground speed in metres per second, the airflow phase —
    // the integral of speed squared, accumulated by the caller a tick at a time — and the car's own
    // two axes in world space, which must be the body's and not the ground's. A per-tick setter
    // where setRain is a per-scene statement, because speed is the car's and changes every tick
    // where the weather does not. See Scene::rainMotion and Scene::rainForward for what each number
    // is, why the integral rides along, and why neither axis may be flattened.
    [[nodiscard]] std::expected<void, std::string> setRainMotion(Scene& scene, float groundSpeed, float airflowPhase,
                                                                 glm::vec3 forward, glm::vec3 bodyUp) const;
    // The blades on the glass the rain lands on. Geometry and a timing law rather than a blade
    // angle, because the angle is the clock's to answer — see Wipers, which is where the whole of
    // why this is a scene statement and not a per-frame stream is written down.
    [[nodiscard]] std::expected<void, std::string> setWipers(Scene& scene, const Wipers& wipers) const;
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
