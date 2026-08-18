module;

#include <cstddef>
#include <cstring>
#include <expected>
#include <string>
#include <utility>
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
    static std::expected<std::vector<T>, std::string> readBytes(const tinygltf::Model& model,
                                                                const tinygltf::Accessor& accessor);

    static size_t componentsPerElement(const tinygltf::Accessor& accessor)
    {
        return static_cast<size_t>(tinygltf::GetNumComponentsInType(static_cast<uint32_t>(accessor.type)));
    }

    // "the accessor read, but it holds fewer components than this type needs" — a distinct
    // condition from a malformed accessor, and one every specialization states the same way. A
    // member rather than a file-local helper because the specializations below are reachable
    // from importers, and a TU-local name is not.
    [[nodiscard]] static std::string tooFew(const size_t held, const size_t needed)
    {
        return "accessor holds " + std::to_string(held) + " component(s), fewer than the " + std::to_string(needed) +
               " this type needs";
    }

public:
    // Only the specializations below are readable; anything else is a caller mistake rather
    // than a runtime failure, so it is rejected at compile time instead of silently
    // returning an indeterminate value.
    //
    // A malformed accessor is the other thing, and it *is* a runtime condition: the file said so.
    // These used to answer it with an empty vector or a zeroed value, which every caller then
    // read as "the asset has no data here" — indistinguishable from an accessor that legitimately
    // describes none, and the reason the file was wrong went nowhere. It is reported now.
    template <typename T>
    [[nodiscard]] inline static std::expected<T, std::string> get(const tinygltf::Model&, const tinygltf::Accessor&)
    {
        static_assert(false, "AccessorUtility::get is only defined for the specializations in this unit");
    }
};

// Every index below comes from the file and glTF allows -1 for an absent one, so each is
// range-checked before use: unchecked, a -1 becomes SIZE_MAX.
template <typename T>
inline std::expected<std::vector<T>, std::string> AccessorUtility::readBytes(const tinygltf::Model& model,
                                                                             const tinygltf::Accessor& accessor)
{
    if (accessor.bufferView < 0 || std::cmp_greater_equal(accessor.bufferView, model.bufferViews.size()))
    {
        return std::unexpected("accessor names buffer view " + std::to_string(accessor.bufferView) + " of " +
                               std::to_string(model.bufferViews.size()));
    }

    const auto& bufferView = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    if (bufferView.buffer < 0 || std::cmp_greater_equal(bufferView.buffer, model.buffers.size()))
    {
        return std::unexpected("buffer view names buffer " + std::to_string(bufferView.buffer) + " of " +
                               std::to_string(model.buffers.size()));
    }

    const auto& buffer = model.buffers[static_cast<size_t>(bufferView.buffer)];

    const auto byteStride = accessor.ByteStride(bufferView);
    const auto componentBytes = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(accessor.componentType));
    if (byteStride <= 0 || componentBytes <= 0)
    {
        return std::unexpected("accessor has byte stride " + std::to_string(byteStride) + " and component size " +
                               std::to_string(componentBytes) + ", neither of which can be zero or negative");
    }

    if (accessor.count == 0)
    {
        return std::vector<T>();
    }

    const auto stride = static_cast<size_t>(byteStride);
    const auto components = componentsPerElement(accessor);
    const auto elementBytes = static_cast<size_t>(componentBytes) * components;
    const auto base = bufferView.byteOffset + accessor.byteOffset;

    // The destination slot per element is components * sizeof(T); a component type wider than
    // T would otherwise overrun it.
    if (elementBytes > components * sizeof(T))
    {
        return std::unexpected("accessor element is " + std::to_string(elementBytes) + " byte(s), wider than the " +
                               std::to_string(components * sizeof(T)) +
                               " byte(s) the destination component type holds");
    }

    // Written as subtraction and division so an accessor claiming an absurd count or offset
    // cannot overflow the arithmetic that is meant to catch it.
    const auto available = buffer.data.size();
    if (base > available || elementBytes > available - base)
    {
        return std::unexpected("accessor starts at byte " + std::to_string(base) + " of a " +
                               std::to_string(available) + " byte buffer and needs " + std::to_string(elementBytes) +
                               " more");
    }

    if (accessor.count > (available - base - elementBytes) / stride + 1)
    {
        return std::unexpected("accessor claims " + std::to_string(accessor.count) + " element(s) of stride " +
                               std::to_string(stride) + ", past the end of its " + std::to_string(available) +
                               " byte buffer");
    }

    auto out = std::vector<T>(accessor.count * components);

    for (size_t i = 0; i < accessor.count; i++)
    {
        std::memcpy(out.data() + i * components, buffer.data.data() + base + i * stride, elementBytes);
    }

    return out;
}

template <>
inline std::expected<float, std::string> AccessorUtility::get<float>(const tinygltf::Model& model,
                                                                     const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (!result)
    {
        return std::unexpected(result.error());
    }

    if (result->empty())
    {
        return std::unexpected(tooFew(result->size(), 1));
    }

    return result->front();
}

template <>
inline std::expected<std::vector<float>, std::string>
AccessorUtility::get<std::vector<float>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    return readBytes<float>(model, accessor);
}

template <>
inline std::expected<glm::vec2, std::string> AccessorUtility::get<glm::vec2>(const tinygltf::Model& model,
                                                                             const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (!result)
    {
        return std::unexpected(result.error());
    }

    if (result->size() < 2)
    {
        return std::unexpected(tooFew(result->size(), 2));
    }

    return glm::make_vec2(result->data());
}

template <>
inline std::expected<glm::vec3, std::string> AccessorUtility::get<glm::vec3>(const tinygltf::Model& model,
                                                                             const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (!result)
    {
        return std::unexpected(result.error());
    }

    if (result->size() < 3)
    {
        return std::unexpected(tooFew(result->size(), 3));
    }

    return glm::make_vec3(result->data());
}

template <>
inline std::expected<std::vector<glm::vec3>, std::string>
AccessorUtility::get<std::vector<glm::vec3>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);

    if (!results)
    {
        return std::unexpected(results.error());
    }

    auto out = std::vector<glm::vec3>();
    out.reserve(results->size() / 3);

    for (size_t i = 0; i + 3 <= results->size(); i += 3)
    {
        out.push_back(glm::make_vec3(results->data() + i));
    }

    return out;
}

template <>
inline std::expected<glm::quat, std::string> AccessorUtility::get<glm::quat>(const tinygltf::Model& model,
                                                                             const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (!result)
    {
        return std::unexpected(result.error());
    }

    if (result->size() < 4)
    {
        return std::unexpected(tooFew(result->size(), 4));
    }

    return glm::make_quat(result->data());
}

template <>
inline std::expected<std::vector<glm::quat>, std::string>
AccessorUtility::get<std::vector<glm::quat>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);

    if (!results)
    {
        return std::unexpected(results.error());
    }

    auto out = std::vector<glm::quat>();
    out.reserve(results->size() / 4);

    for (size_t i = 0; i + 4 <= results->size(); i += 4)
    {
        out.push_back(glm::make_quat(results->data() + i));
    }

    return out;
}

template <>
inline std::expected<glm::vec4, std::string> AccessorUtility::get<glm::vec4>(const tinygltf::Model& model,
                                                                             const tinygltf::Accessor& accessor)
{
    const auto result = readBytes<float>(model, accessor);

    if (!result)
    {
        return std::unexpected(result.error());
    }

    if (result->size() < 4)
    {
        return std::unexpected(tooFew(result->size(), 4));
    }

    return glm::make_vec4(result->data());
}

template <>
inline std::expected<std::vector<glm::mat4>, std::string>
AccessorUtility::get<std::vector<glm::mat4>>(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto results = readBytes<float>(model, accessor);

    if (!results)
    {
        return std::unexpected(results.error());
    }

    auto out = std::vector<glm::mat4>();
    out.reserve(results->size() / 16);

    for (size_t i = 0; i + 16 <= results->size(); i += 16)
    {
        out.push_back(glm::make_mat4(results->data() + i));
    }

    return out;
}

} // namespace raceengine
