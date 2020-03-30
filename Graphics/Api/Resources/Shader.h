#pragma once

#include <string>

class Shader
{
public:
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    std::string tessellationControlShaderSource;
    std::string tessellationEvaluationShaderSource;
    std::string computeShaderSource;
    std::string geometryShaderSource;

    explicit Shader()
    {};
};
