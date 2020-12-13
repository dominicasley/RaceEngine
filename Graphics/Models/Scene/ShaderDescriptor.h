#pragma once

#include <string>

struct ShaderDescriptor
{
public:
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    std::string tessellationControlShaderSource;
    std::string tessellationEvaluationShaderSource;
    std::string computeShaderSource;
    std::string geometryShaderSource;
};
