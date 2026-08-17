module;

#include <string>

export module raceengine.graphics.models:Shader;

namespace raceengine
{

export struct Shader {
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
};

} // namespace raceengine
