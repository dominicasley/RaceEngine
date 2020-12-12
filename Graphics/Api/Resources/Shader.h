#pragma once

#include <string>

struct Shader
{
public:
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    std::string tessellationControlShaderSource;
    std::string tessellationEvaluationShaderSource;
    std::string computeShaderSource;
    std::string geometryShaderSource;
};
