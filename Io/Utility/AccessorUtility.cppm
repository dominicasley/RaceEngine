module;

#include <cstddef>
#include <cstring>
#include <vector>

#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <tiny_gltf.h>

export module raceengine.io:AccessorUtility;

namespace raceengine
{

export class AccessorUtility
{
private:
    template <typename T>
    static std::vector<T> readBytes(const tinygltf::Model& model, const tinygltf::Accessor& accessor);

    static size_t componentsPerElement(const tinygltf::Accessor& accessor)
    {
        return static_cast<size_t>(tinygltf::GetNumComponentsInType(static_cast<uint32_t>(accessor.type)));
    }

public:
    template <typename T> inline static T get(const tinygltf::Model&, const tinygltf::Accessor&) {};
};

template <typename T>
inline std::vector<T> AccessorUtility::readBytes(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto& bufferView = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    const auto& buffer = model.buffers[static_cast<size_t>(bufferView.buffer)];

    const auto stride = static_cast<size_t>(accessor.ByteStride(bufferView));
    const auto components = componentsPerElement(accessor);
    const auto elementBytes =
        static_cast<size_t>(tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(accessor.componentType))) *
        components;
    const auto base = bufferView.byteOffset + accessor.byteOffset;

    auto out = std::vector<T>(accessor.count * components);

    for (size_t i = 0; i < accessor.count; i++)
    {
        std::memcpy(out.data() + i * components, buffer.data.data() + base + i * stride, elementBytes);
    }

    return out;
}

template <> inline float AccessorUtility::get<float>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (result.empty())
    {
        return 0.0f;
    }

    return result[0];
}

template <>
inline std::vector<float> AccessorUtility::get<std::vector<float>>(const tinygltf::Model& model,
                                                                   const tinygltf::Accessor& accessor)
{
    return readBytes<float>(model, accessor);
}

template <> inline glm::vec2 AccessorUtility::get<glm::vec2>(const tinygltf::Model&, const tinygltf::Accessor&)
{
    return glm::vec2(1.0f);
}

template <>
inline glm::vec3 AccessorUtility::get<glm::vec3>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (result.size() < 3)
    {
        return glm::vec3(0.0f);
    }

    return glm::make_vec3(result.data());
}

template <>
inline std::vector<glm::vec3> AccessorUtility::get<std::vector<glm::vec3>>(const tinygltf::Model& model,
                                                                           const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);
    auto out = std::vector<glm::vec3>();
    out.reserve(results.size() / 3);

    for (size_t i = 0; i + 3 <= results.size(); i += 3)
    {
        out.push_back(glm::make_vec3(results.data() + i));
    }

    return out;
}

template <>
inline glm::quat AccessorUtility::get<glm::quat>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (result.size() < 4)
    {
        return glm::quat();
    }

    return glm::make_quat(result.data());
}

template <>
inline std::vector<glm::quat> AccessorUtility::get<std::vector<glm::quat>>(const tinygltf::Model& model,
                                                                           const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);
    auto out = std::vector<glm::quat>();
    out.reserve(results.size() / 4);

    for (size_t i = 0; i + 4 <= results.size(); i += 4)
    {
        out.push_back(glm::make_quat(results.data() + i));
    }

    return out;
}

template <> inline glm::vec4 AccessorUtility::get<glm::vec4>(const tinygltf::Model&, const tinygltf::Accessor&)
{
    return glm::vec4(1.0f);
}

template <>
inline std::vector<glm::mat4> AccessorUtility::get<std::vector<glm::mat4>>(const tinygltf::Model& model,
                                                                           const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);
    auto out = std::vector<glm::mat4>();
    out.reserve(results.size() / 16);

    for (size_t i = 0; i + 16 <= results.size(); i += 16)
    {
        out.push_back(glm::make_mat4(results.data() + i));
    }

    return out;
}

} // namespace raceengine
