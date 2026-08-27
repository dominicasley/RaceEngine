// ShadowService bodies. Declarations are in Graphics/Services/ShadowService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <string>
#include <utility>

#include <glm/glm.hpp>

module raceengine.graphics;

import :ShadowService;
import :CameraService;
import :RenderContract;
import :ShadowCascades;
import raceengine.graphics.models;

namespace raceengine
{

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
        // The far half of the map may run coarser — see CreateShadowCascadesDTO::farResolution.
        const auto resolution = index >= 2 && createShadowCascadesDTO.farResolution != 0
                                    ? createShadowCascadesDTO.farResolution
                                    : createShadowCascadesDTO.resolution;

        // Depth only, at its own resolution, with the comparison sampler percentage-closer
        // filtering needs — and marked as a producer, which is what makes the frame record it
        // before the camera that samples it.
        auto camera =
            cameraService.createCamera(CreateCameraDTO{.width = resolution,
                                                       .height = resolution,
                                                       .target = CameraTarget::DepthOnly,
                                                       .depthComparison = DepthComparison::LessOrEqual,
                                                       .role = CameraRole::ShadowCascade,
                                                       .debugName = "cascade " + std::to_string(index),
                                                       .overrideShader = createShadowCascadesDTO.depthShader});
        if (!camera)
        {
            // The cascades built so far are still in the scene's deque and would render into
            // targets nothing reads, so the map is abandoned whole rather than left partial.
            scene.shadows.cascades.clear();

            return std::unexpected("cascade " + std::to_string(index) + " has no depth target: " + camera.error());
        }

        scene.shadows.cascades.push_back(ShadowCascade{
            .camera = &scene.cameras.emplace_back(std::move(camera).value()), .resolution = resolution});
    }

    scene.shadows.light = &light;
    scene.shadows.viewCamera = &viewCamera;
    scene.shadows.lightIndex = lightIndex;
    scene.shadows.resolution = createShadowCascadesDTO.resolution;
    scene.shadows.cacheFarCascades = createShadowCascadesDTO.cacheFarCascades;
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

        // A cached far cascade pads its fit so the picture in it survives the camera drifting,
        // and refits — re-rendering with it — only when the ideal fit leaves the pad. The pad is
        // a fraction of the slice's own radius, so every cascade's hold budget scales with what
        // it covers; the safety margin under it is what keeps the original slice inside the
        // padded square right up to the refit.
        constexpr auto farCascadePadFraction = 0.15f;
        constexpr auto refitSafety = 0.9f;
        const auto cached = scene.shadows.cacheFarCascades && index >= 2;
        const auto pad = cached ? farCascadePadFraction *
                                      cascadeSliceSphere(viewCamera.fieldOfView, viewCamera.aspectRatio,
                                                         splits[index], splits[index + 1])
                                          .radius
                                : 0.0f;

        // The cascade's own size, never the shared one: the texel arithmetic the bias budget is
        // stated in has to divide by the map actually being rendered.
        const auto resolution = cascade.resolution != 0 ? cascade.resolution : scene.shadows.resolution;
        const auto fit = fitCascade(viewCamera.position, viewCamera.direction, viewCamera.fieldOfView,
                                    viewCamera.aspectRatio, splits[index], splits[index + 1],
                                    scene.shadows.light->direction, resolution, scene.shadows.casterExtent, pad);

        auto& camera = *cascade.camera;

        if (cached && cascade.mapValid && glm::length(fit.position - cascade.heldPosition) < pad * refitSafety)
        {
            camera.contentsHeld = true;

            continue;
        }

        cameraService.setPosition(camera, fit.position.x, fit.position.y, fit.position.z);
        cameraService.setDirection(camera, fit.direction.x, fit.direction.y, fit.direction.z);
        cameraService.setRoll(camera, fit.roll.x, fit.roll.y, fit.roll.z);
        cameraService.setClippingPlanes(camera, fit.nearPlane, fit.farPlane);
        cameraService.setOrthographic(camera, fit.volume.left, fit.volume.right, fit.volume.bottom, fit.volume.top);

        cascade.splitDistance = splits[index + 1];
        cascade.texelWorldSize = fit.texelWorldSize;
        cascade.depthPerWorldUnit = fit.depthPerWorldUnit;
        cascade.heldPosition = fit.position;
        cascade.mapValid = cached;
        camera.contentsHeld = false;
    }
}

} // namespace raceengine
