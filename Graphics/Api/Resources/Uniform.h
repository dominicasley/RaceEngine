#pragma once

#include <any>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

#include "UniformDataType.h"


class Uniform
{
public:
    const std::any data;
    const UniformDataType dataType;

    Uniform();

    ~Uniform();

    Uniform operator=(const Uniform &uniform)
    {
        return Uniform(uniform.data, uniform.dataType);
    };

    explicit Uniform(std::any value, UniformDataType type) : data(value), dataType(type)
    {};

    explicit Uniform(const int &value) : data(std::reference_wrapper<const int>(value)),
                                         dataType(UniformDataType(value))
    {};

    explicit Uniform(const float &value) : data(std::reference_wrapper<const float>(value)),
                                           dataType(UniformDataType(value))
    {};

    explicit Uniform(const double &value) : data(std::reference_wrapper<const double>(value)),
                                            dataType(UniformDataType(value))
    {};

    explicit Uniform(const glm::vec2 &value) : data(std::reference_wrapper<const glm::vec2>(value)),
                                               dataType(UniformDataType(value))
    {};

    explicit Uniform(const glm::vec3 &value) : data(std::reference_wrapper<const glm::vec3>(value)),
                                               dataType(UniformDataType(value))
    {};

    explicit Uniform(const glm::vec4 &value) : data(std::reference_wrapper<const glm::vec4>(value)),
                                               dataType(UniformDataType(value))
    {};

    explicit Uniform(const glm::mat3 &value) : data(std::reference_wrapper<const glm::mat3>(value)),
                                               dataType(UniformDataType(value))
    {};

    explicit Uniform(const glm::mat4 &value) : data(std::reference_wrapper<const glm::mat4>(value)),
                                               dataType(UniformDataType(value))
    {};

    explicit Uniform(const std::vector<glm::vec2> &value) : data(
            std::reference_wrapper<const std::vector<glm::vec2>>(value)), dataType(UniformDataType(value))
    {};

    explicit Uniform(const std::vector<glm::vec3> &value) : data(
            std::reference_wrapper<const std::vector<glm::vec3>>(value)), dataType(UniformDataType(value))
    {};

    explicit Uniform(const std::vector<glm::vec4> &value) : data(
            std::reference_wrapper<const std::vector<glm::vec4>>(value)), dataType(UniformDataType(value))
    {};

    explicit Uniform(const std::vector<glm::mat3> &value) : data(
            std::reference_wrapper<const std::vector<glm::mat3>>(value)), dataType(UniformDataType(value))
    {};

    explicit Uniform(const std::vector<glm::mat4> &value) : data(
            std::reference_wrapper<const std::vector<glm::mat4>>(value)), dataType(UniformDataType(value))
    {};

    template<typename T>
    const T &value()
    {
        return std::any_cast<std::reference_wrapper<const T>>(data).get();
    }

    UniformDataTypes::Type type() const
    {
        return this->dataType.type();
    }
};

inline Uniform::Uniform() : dataType(UniformDataType(0))
{

}

inline Uniform::~Uniform() = default;

