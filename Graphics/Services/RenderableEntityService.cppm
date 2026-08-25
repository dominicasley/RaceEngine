module;

#include <cmath>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/span.h>

export module raceengine.graphics:RenderableEntityService;

import :FrameDiagnostics;
import :ShaderService;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class RenderableEntityService
{
private:
    MemoryStorageService& memoryStorageService;
    // joints() runs inside the draw path and has nowhere to report to; a clip ozz refuses to
    // sample is counted here rather than becoming an empty palette with no explanation.
    FrameDiagnostics& diagnostics;

public:
    explicit RenderableEntityService(MemoryStorageService& memoryStorageService, FrameDiagnostics& diagnostics);

    // `shaderService` is taken as a parameter rather than held, and that is the composition root's
    // doing rather than a preference: this service is constructed *before* the renderer, and the
    // shader registry is constructed after it, so there is no reference to hold at the time this one
    // is built. Passing it in at the one call site keeps the whole material-shader decision in one
    // place instead of splitting it across two services to work around an ordering.
    [[nodiscard]] RenderableModel createModel(const CreateRenderableModelDTO& entityDescriptor,
                                              ShaderService& shaderService) const;
    [[nodiscard]] std::expected<void, std::string>
    setSkeleton(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Skeleton>> skeleton) const;
    [[nodiscard]] std::expected<void, std::string>
    addAnimation(RenderableMesh& mesh, Resource<std::unique_ptr<ozz::animation::Animation>> animation) const;
    [[nodiscard]] std::expected<void, std::string> setAnimation(RenderableMesh& mesh,
                                                                const std::string& animationName) const;
    [[nodiscard]] std::expected<void, std::string> setAnimation(RenderableMesh& mesh,
                                                                unsigned int animationIndex) const;
    // The palette lives on the mesh and is rebuilt in place; it stays valid until this mesh's
    // next joints() call.
    [[nodiscard]] const std::vector<glm::mat4>& joints(RenderableMesh& renderableMesh, float frameTimeDelta) const;

private:
    // For messages only, so a handle whose mesh has been unloaded names itself rather than
    // throwing out of the reporting path of an error that has already happened.
    [[nodiscard]] std::string meshName(const RenderableMesh& mesh) const;
};

glm::mat4 ozzToMat4(const ozz::math::Float4x4& t);

} // namespace raceengine
