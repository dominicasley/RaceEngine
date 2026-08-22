// ShaderService bodies. Declarations are in Graphics/Services/ShaderService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <optional>
#include <string>

module raceengine.graphics;

import :ShaderService;
import :IGpuResourceFactory;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

ShaderService::ShaderService(MemoryStorageService& memoryStorageService, IGpuResourceFactory& gpuResourceFactory) :
    memoryStorageService(memoryStorageService),
    gpuResourceFactory(gpuResourceFactory)
{
}

std::expected<Resource<Shader>, std::string> ShaderService::createShader(const std::string& name,
                                                                         const ShaderDescriptor& shaderDescriptor)
{
    const auto shaderProgramId = gpuResourceFactory.createShaderObject(shaderDescriptor);

    if (!shaderProgramId)
    {
        return std::unexpected("shader '" + name + "' was not created: " + shaderProgramId.error());
    }

    const auto shader = memoryStorageService.shaders.add(Shader{.gpuResourceId = shaderProgramId.value()});

    shaders[name] = shader;

    return shader;
}

std::optional<Resource<Shader>> ShaderService::getShaderByName(const std::string& name)
{
    if (shaders.contains(name))
        return shaders.at(name);

    return {};
}

} // namespace raceengine
