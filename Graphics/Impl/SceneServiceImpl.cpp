// SceneService bodies. Declarations are in Graphics/Services/SceneService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <functional>
#include <string>
#include <utility>

#include <glm/glm.hpp>

module raceengine.graphics;

import :SceneService;
import :RenderableEntityService;
import :CameraService;
import :SceneManagerService;
import :ShaderService;
import raceengine.graphics.models;

namespace raceengine
{

SceneService::SceneService(RenderableEntityService& renderableEntityService, CameraService& cameraService,
                           SceneManagerService& sceneManagerService, ShaderService& shaderService) :
    renderableEntityService(renderableEntityService),
    cameraService(cameraService),
    sceneManagerService(sceneManagerService),
    shaderService(shaderService)
{
}

void SceneService::update(Scene& scene, float) const
{
    for (auto& node : scene.nodes)
    {
        static_cast<void>(sceneManagerService.modelMatrix(node));
    }
}

RenderableModel& SceneService::createEntity(Scene& scene, const CreateRenderableModelDTO& entityDescriptor) const
{
    return scene.models.emplace_back(renderableEntityService.createModel(entityDescriptor, shaderService));
}

std::expected<std::reference_wrapper<Camera>, std::string> SceneService::createCamera(Scene& scene) const
{
    auto camera = cameraService.createCamera();
    if (!camera)
    {
        return std::unexpected(camera.error());
    }

    return std::ref(scene.cameras.emplace_back(std::move(camera).value()));
}

std::expected<std::reference_wrapper<Camera>, std::string>
SceneService::createCamera(Scene& scene, const CreateCameraDTO& createCameraDTO) const
{
    auto camera = cameraService.createCamera(createCameraDTO);
    if (!camera)
    {
        return std::unexpected(camera.error());
    }

    return std::ref(scene.cameras.emplace_back(std::move(camera).value()));
}

Light& SceneService::createLight(Scene& scene) const
{
    return scene.lights.emplace_back(Light());
}

Camera& SceneService::getCamera(Scene& scene, unsigned int index) const
{
    return scene.cameras.at(index);
}

RenderableModel& SceneService::getModel(Scene& scene, unsigned int index) const
{
    return scene.models.at(index);
}

std::expected<void, std::string> SceneService::setFog(Scene& scene, const Fog& fog) const
{
    if (fog.density < 0.0f)
    {
        return std::unexpected("fog density is an extinction coefficient and cannot be negative");
    }

    if (fog.scaleHeight <= 0.0f)
    {
        return std::unexpected("fog needs a scale height above zero: it is the height the density falls by 1/e over, "
                               "and zero is a layer with no thickness rather than a layer of no fog");
    }

    if (fog.maximumDistance <= 0.0f)
    {
        return std::unexpected("fog needs a maximum distance above zero to integrate the medium along");
    }

    // The published function is undefined at exactly one and this engine clamps rather than reaching
    // it, so the bound refused here is the one that means something different from what it says: a
    // medium that scatters *all* of the light in one direction is a mirror, not fog.
    if (fog.anisotropy <= -1.0f || fog.anisotropy >= 1.0f)
    {
        return std::unexpected("fog anisotropy is Henyey-Greenstein's asymmetry and lies strictly between -1 and 1");
    }

    // A single-scatter albedo is the fraction of extinction that scatters rather than absorbs, so
    // above one the medium hands back more light than reached it and the fog glows on its own.
    if (glm::any(glm::lessThan(fog.scatteringAlbedo, glm::vec3(0.0f))) ||
        glm::any(glm::greaterThan(fog.scatteringAlbedo, glm::vec3(1.0f))))
    {
        return std::unexpected("a fog scattering albedo is a fraction of the extinction and lies between 0 and 1");
    }

    scene.fog = fog;

    return {};
}

std::expected<void, std::string> SceneService::setRain(Scene& scene, const float rain) const
{
    if (rain < 0.0f)
    {
        return std::unexpected("rain is an intensity and cannot be negative; zero is the dry scene");
    }

    scene.rain = rain;

    return {};
}

std::expected<void, std::string> SceneService::setClouds(Scene& scene, const float coverage, const float type) const
{
    if (coverage < 0.0f)
    {
        return std::unexpected("cloud coverage is an amount of sky and cannot be negative; zero is the clear sky");
    }

    // A blend, unlike coverage: past either end there is no cloud kind it names.
    if (type < 0.0f || type > 1.0f)
    {
        return std::unexpected("cloud type is a stratus-to-cumulus blend and lies between 0 and 1");
    }

    scene.clouds = Clouds{coverage, type};

    return {};
}

void SceneService::setCloudMap(Scene& scene, const Resource<FboAttachment>& cloudMap) const
{
    scene.cloudMap = cloudMap;
}

std::expected<void, std::string> SceneService::setRainMotion(Scene& scene, const float groundSpeed,
                                                             const float airflowPhase, const glm::vec3 forward,
                                                             const glm::vec3 bodyUp) const
{
    if (groundSpeed < 0.0f)
    {
        return std::unexpected("ground speed is a magnitude and cannot be negative");
    }

    // Both are directions and a direction of no length is not one. Refused rather than defaulted:
    // a zero here would leave the shader normalising nothing, and every drop on the car would take
    // its heading from a NaN.
    if (glm::dot(forward, forward) < 1.0e-8f || glm::dot(bodyUp, bodyUp) < 1.0e-8f)
    {
        return std::unexpected("the car's forward and up are directions and cannot be zero length");
    }

    scene.rainMotion = glm::vec2(groundSpeed, airflowPhase);
    scene.rainForward = glm::normalize(forward);
    scene.rainBodyUp = glm::normalize(bodyUp);

    return {};
}

std::expected<void, std::string> SceneService::setWipers(Scene& scene, const Wipers& wipers) const
{
    // Zero is the car with none, so only a *running* pair has anything to be wrong about.
    if (wipers.cyclePeriod > 0.0f)
    {
        if (wipers.sweepSeconds <= 0.0f || wipers.sweepSeconds > wipers.cyclePeriod)
        {
            return std::unexpected("a wiper's sweep must take some time and cannot outlast its own cycle");
        }

        if (wipers.bladeHalfWidth < 0.0f)
        {
            return std::unexpected("a wiper blade's half width is a distance and cannot be negative");
        }

        // An inner radius past the outer is an arc that contains nothing, which reads as wipers that
        // are running and clearing not one drop — the failure this sentence exists to name.
        if (wipers.bladeA.z > wipers.bladeA.w || wipers.bladeB.z > wipers.bladeB.w)
        {
            return std::unexpected("a wiper blade's inner radius cannot be past its outer");
        }

        if (wipers.sweepAngle.x == 0.0f && wipers.sweepAngle.y == 0.0f)
        {
            return std::unexpected("wipers that sweep through no angle at all clear nothing; zero the period instead");
        }
    }

    scene.wipers = wipers;

    return {};
}

} // namespace raceengine
