#pragma once

#include <vector>
#include <tiny_gltf.h>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>


class AccessorUtility
{
private:
    template<typename T>
    static std::vector<std::vector<T>> readBytes(const tinygltf::Model& model, const tinygltf::Accessor& accessor);

public:
    template<typename T>
    inline static T get(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
    {};
};

template<typename T> inline
std::vector<std::vector<T>> AccessorUtility::readBytes(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    auto accessorOffset = accessor.byteOffset;
    auto stride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
    auto offset = model.bufferViews[accessor.bufferView].byteOffset;

    auto out = std::vector<std::vector<T>>(accessor.count);
    const auto buffer = model.buffers[model.bufferViews[accessor.bufferView].buffer];

    for (auto i = 0; i < accessor.count; i++)
    {
        auto start = buffer.data.begin() + accessorOffset + offset + i * stride;
        auto chunk = std::vector<unsigned char>(start, start +
                                                       tinygltf::GetComponentSizeInBytes(accessor.componentType) *
                                                       tinygltf::GetNumComponentsInType(accessor.type));

        auto typedChunk = std::vector<T>(tinygltf::GetNumComponentsInType(accessor.type));
        memcpy(typedChunk.data(), chunk.data(),
               static_cast<size_t>(tinygltf::GetComponentSizeInBytes(accessor.componentType) *
                                   tinygltf::GetNumComponentsInType(accessor.type)));

        out[i] = typedChunk;
    }

    return out;
}


template<> inline
float AccessorUtility::get<float>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (result.empty())
    {
        return 0.0f;
    }

    return result[0][0];
}

template<> inline
std::vector<float>
AccessorUtility::get<std::vector<float>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);
    auto out = std::vector<float>();

    for (auto floatSet : results)
    {
        for (auto f : floatSet)
        {
            out.push_back(f);
        }
    }

    return out;
}

template<> inline
glm::vec2 AccessorUtility::get<glm::vec2>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    return glm::vec2(1.0f);
}

template<> inline
glm::vec3 AccessorUtility::get<glm::vec3>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (result.empty())
    {
        return glm::vec3(0.0f);
    }

    return glm::make_vec3(result[0].data());
}

template<> inline
std::vector<glm::vec3>
AccessorUtility::get<std::vector<glm::vec3>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);
    auto out = std::vector<glm::vec3>();

    for (auto vectorSet : results)
    {
        out.push_back(glm::make_vec3(vectorSet.data()));
    }

    return out;
}

template<> inline
glm::quat AccessorUtility::get<glm::quat>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (result.empty())
    {
        return glm::quat();
    }

    return glm::make_quat(result[0].data());
}

template<> inline
std::vector<glm::quat>
AccessorUtility::get<std::vector<glm::quat>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);
    auto out = std::vector<glm::quat>();

    for (auto m4 : results)
    {
        out.push_back(glm::make_quat(m4.data()));
    }

    return out;
}

template<> inline
glm::vec4 AccessorUtility::get<glm::vec4>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    return glm::vec4(1.0f);
}

template<> inline
std::vector<glm::mat4>
AccessorUtility::get<std::vector<glm::mat4>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);
    auto out = std::vector<glm::mat4>();

    for (auto m4 : results)
    {
        out.push_back(glm::make_mat4(m4.data()));
    }

    return out;
}
