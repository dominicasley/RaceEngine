#pragma once

#include <typeindex>
#include <unordered_map>

namespace UniformDataTypes
{
    enum Type
    {
        INT,
        FLOAT,
        DOUBLE,
        VEC2,
        VEC3,
        VEC4,
        MAT3,
        MAT4,
        VECTOR_OF_VEC2,
        VECTOR_OF_VEC3,
        VECTOR_OF_VEC4,
        VECTOR_OF_MAT3,
        VECTOR_OF_MAT4,
        UNSUPPORTED,
    };
}


class UniformDataType
{
private:
    const UniformDataTypes::Type dataType;

public:
    static const inline std::unordered_map<std::type_index, UniformDataTypes::Type> map{
            {typeid(const int &),                    UniformDataTypes::INT},
            {typeid(const float &),                  UniformDataTypes::FLOAT},
            {typeid(const double &),                 UniformDataTypes::DOUBLE},
            {typeid(const glm::vec2 &),              UniformDataTypes::VEC2},
            {typeid(const glm::vec3 &),              UniformDataTypes::VEC3},
            {typeid(const glm::vec4 &),              UniformDataTypes::VEC4},
            {typeid(const glm::mat3 &),              UniformDataTypes::MAT3},
            {typeid(const glm::mat4 &),              UniformDataTypes::MAT4},
            {typeid(const std::vector<glm::vec2> &), UniformDataTypes::VECTOR_OF_VEC2},
            {typeid(const std::vector<glm::vec3> &), UniformDataTypes::VECTOR_OF_VEC3},
            {typeid(const std::vector<glm::vec4> &), UniformDataTypes::VECTOR_OF_VEC4},
            {typeid(const std::vector<glm::mat3> &), UniformDataTypes::VECTOR_OF_MAT3},
            {typeid(const std::vector<glm::mat4> &), UniformDataTypes::VECTOR_OF_MAT4},
    };

    template<typename T>
    explicit UniformDataType(const T &data) : dataType(UniformDataType::map.at(typeid(data)))
    {
    }

    UniformDataTypes::Type type() const
    {
        return dataType;
    }
};

