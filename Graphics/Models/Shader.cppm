module;

#include <string>
#include <utility>
#include <vector>

export module raceengine.graphics.models:Shader;

namespace raceengine
{

export struct Shader
{
    unsigned int gpuResourceId;
};

export struct ShaderDescriptor
{
public:
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    std::string tessellationControlShaderSource{};
    std::string tessellationEvaluationShaderSource{};
    std::string computeShaderSource{};
    std::string geometryShaderSource{};
    // Preprocessor definitions this shader is compiled with, beside the contract's own: a name and a
    // value, exactly as `#define name value` would state them. This is how one source is registered
    // twice as two shaders — the police light bar's lens is `PbrFragmentShader.glsl` with
    // `BEACON_LENS` defined — without a second copy of a thousand lines that must not drift from
    // the first. They join the SPIR-V cache key, so two registrations of one source are two entries.
    std::vector<std::pair<std::string, std::string>> defines{};
};

} // namespace raceengine
