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

ShadowService::ShadowService(CameraService& cameraService) :
    cameraService(cameraService)
{
}

std::expected<void, std::string> ShadowService::enable(Scene& scene, Light& light, Camera& viewCamera,
                                                       const CreateShadowCascadesDTO& createShadowCascadesDTO) const
{
    if (!scene.shadows.cascades.empty())
    {
        return std::unexpected("the scene already has a cascaded shadow map");
    }

    if (light.type != LightType::Directional)
    {
        return std::unexpected("a cascaded shadow map splits a directional light's frustum; this light is a point "
                               "light and has no direction to split along");
    }

    if (glm::length(light.direction) <= 0.0f)
    {
        return std::unexpected("the directional light has no direction, so there is nothing to cast along");
    }

    // Which light the shaders shadow. The backends upload Scene::lights in order and clamp at
    // maxLights, so a light past the cap is uploaded by nobody and would be shadowed by nobody.
    auto lightIndex = 0u;
    auto found = false;
    for (const auto& candidate : scene.lights)
    {
        if (&candidate == &light)
        {
            found = true;
            break;
        }

        lightIndex++;
    }

    if (!found)
    {
        return std::unexpected("the light does not belong to this scene");
    }

    if (lightIndex >= maxLights)
    {
        return std::unexpected("the light sits at index " + std::to_string(lightIndex) + ", past the " +
                               std::to_string(maxLights) + " the shaders carry");
    }

    scene.shadows.cascades.reserve(shadowCascadeCount);

    for (auto index = 0u; index < shadowCascadeCount; index++)
    {
        // Depth only, at its own resolution, with the comparison sampler percentage-closer
        // filtering needs — and marked as a producer, which is what makes the frame record it
        // before the camera that samples it.
        auto camera =
            cameraService.createCamera(CreateCameraDTO{.width = createShadowCascadesDTO.resolution,
                                                       .height = createShadowCascadesDTO.resolution,
                                                       .target = CameraTarget::DepthOnly,
                                                       .depthComparison = DepthComparison::LessOrEqual,
                                                       .role = CameraRole::ShadowCascade,
                                                       .overrideShader = createShadowCascadesDTO.depthShader});
        if (!camera)
        {
            // The cascades built so far are still in the scene's deque and would render into
            // targets nothing reads, so the map is abandoned whole rather than left partial.
            scene.shadows.cascades.clear();

            return std::unexpected("cascade " + std::to_string(index) + " has no depth target: " + camera.error());
        }

        scene.shadows.cascades.push_back(
            ShadowCascade{.camera = &scene.cameras.emplace_back(std::move(camera).value())});
    }

    scene.shadows.light = &light;
    scene.shadows.viewCamera = &viewCamera;
    scene.shadows.lightIndex = lightIndex;
    scene.shadows.resolution = createShadowCascadesDTO.resolution;
    scene.shadows.distance = createShadowCascadesDTO.distance;
    scene.shadows.lambda = createShadowCascadesDTO.lambda;
    scene.shadows.casterExtent = createShadowCascadesDTO.casterExtent;

    // Fitted here as well as per frame: a cascade camera whose volume is still the unit cube would
    // spend the frame between construction and the first update drawing the world into a 2x2x2 box.
    update(scene);

    return {};
}

void ShadowService::update(Scene& scene) const
{
    if (scene.shadows.cascades.empty() || scene.shadows.light == nullptr || scene.shadows.viewCamera == nullptr)
    {
        return;
    }

    const auto& viewCamera = *scene.shadows.viewCamera;
    const auto splits =
        cascadeSplitDistances(viewCamera.nearClippingPlane, scene.shadows.distance, scene.shadows.lambda);

    for (auto index = 0u; index < scene.shadows.cascades.size(); index++)
    {
        auto& cascade = scene.shadows.cascades[index];
        if (cascade.camera == nullptr)
        {
            continue;
        }

        const auto fit = fitCascade(
            viewCamera.position, viewCamera.direction, viewCamera.fieldOfView, viewCamera.aspectRatio, splits[index],
            splits[index + 1], scene.shadows.light->direction, scene.shadows.resolution, scene.shadows.casterExtent);

        auto& camera = *cascade.camera;
        cameraService.setPosition(camera, fit.position.x, fit.position.y, fit.position.z);
        cameraService.setDirection(camera, fit.direction.x, fit.direction.y, fit.direction.z);
        cameraService.setRoll(camera, fit.roll.x, fit.roll.y, fit.roll.z);
        cameraService.setClippingPlanes(camera, fit.nearPlane, fit.farPlane);
        cameraService.setOrthographic(camera, fit.volume.left, fit.volume.right, fit.volume.bottom, fit.volume.top);

        cascade.splitDistance = splits[index + 1];
        cascade.texelWorldSize = fit.texelWorldSize;
        cascade.depthPerWorldUnit = fit.depthPerWorldUnit;
    }
}

} // namespace raceengine
