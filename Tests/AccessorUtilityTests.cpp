#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <tiny_gltf.h>

import raceengine.io.accessor;

using Catch::Approx;
using raceengine::AccessorUtility;

namespace
{

// A whole glTF model built by hand, which is the only way to reach the failure arms: a file that
// says buffer view -1 is legal glTF and no asset in this repo contains one, so nothing else in
// the tree has ever run these paths.
struct Fixture
{
    tinygltf::Model model;
    tinygltf::Accessor accessor;
};

Fixture floatAccessor(const std::vector<float>& values, const int type, const size_t count)
{
    Fixture fixture;

    tinygltf::Buffer buffer;
    buffer.data.resize(values.size() * sizeof(float));
    if (!values.empty())
    {
        std::memcpy(buffer.data.data(), values.data(), buffer.data.size());
    }
    fixture.model.buffers.push_back(buffer);

    tinygltf::BufferView bufferView;
    bufferView.buffer = 0;
    bufferView.byteOffset = 0;
    bufferView.byteLength = fixture.model.buffers.front().data.size();
    bufferView.byteStride = 0;
    fixture.model.bufferViews.push_back(bufferView);

    fixture.accessor.bufferView = 0;
    fixture.accessor.byteOffset = 0;
    fixture.accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    fixture.accessor.type = type;
    fixture.accessor.count = count;

    return fixture;
}

} // namespace

TEST_CASE("a well-formed vec3 accessor reads the values the buffer holds", "[accessor]")
{
    const auto fixture = floatAccessor({1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);

    const auto read = AccessorUtility::get<glm::vec3>(fixture.model, fixture.accessor);

    REQUIRE(read.has_value());
    REQUIRE(read->x == Approx(1.0));
    REQUIRE(read->y == Approx(2.0));
    REQUIRE(read->z == Approx(3.0));
}

TEST_CASE("a mat4 accessor reads column-major, which is what the skin path assumes", "[accessor]")
{
    // Two matrices: the identity, then a translation by (7, 8, 9). glm is column-major and so is
    // glTF, so the translation lands in the last four floats of each element.
    std::vector<float> values{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 7.0f, 8.0f, 9.0f, 1.0f};

    const auto fixture = floatAccessor(values, TINYGLTF_TYPE_MAT4, 2);

    const auto read = AccessorUtility::get<std::vector<glm::mat4>>(fixture.model, fixture.accessor);

    REQUIRE(read.has_value());
    REQUIRE(read->size() == 2);
    REQUIRE(read->at(0) == glm::mat4(1.0f));
    REQUIRE(read->at(1)[3].x == Approx(7.0));
    REQUIRE(read->at(1)[3].y == Approx(8.0));
    REQUIRE(read->at(1)[3].z == Approx(9.0));
}

TEST_CASE("a strided accessor skips the padding between elements", "[accessor]")
{
    // Two vec3s in slots 24 bytes apart, with six floats of interleaved something-else between
    // them. A reader that walked the buffer densely would return the padding.
    auto fixture = floatAccessor({1.0f, 2.0f, 3.0f, -1.0f, -1.0f, -1.0f, 4.0f, 5.0f, 6.0f, -1.0f, -1.0f, -1.0f},
                                 TINYGLTF_TYPE_VEC3, 2);
    fixture.model.bufferViews.front().byteStride = 24;

    const auto read = AccessorUtility::get<std::vector<glm::vec3>>(fixture.model, fixture.accessor);

    REQUIRE(read.has_value());
    REQUIRE(read->size() == 2);
    REQUIRE(read->at(0).x == Approx(1.0));
    REQUIRE(read->at(1).x == Approx(4.0));
    REQUIRE(read->at(1).z == Approx(6.0));
}

TEST_CASE("the buffer view's byte offset and the accessor's are both applied", "[accessor]")
{
    auto fixture = floatAccessor({9.0f, 9.0f, 9.0f, 1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);
    fixture.model.bufferViews.front().byteOffset = 4;
    fixture.accessor.byteOffset = 8;

    const auto read = AccessorUtility::get<glm::vec3>(fixture.model, fixture.accessor);

    REQUIRE(read.has_value());
    REQUIRE(read->x == Approx(1.0));
    REQUIRE(read->z == Approx(3.0));
}

TEST_CASE("a buffer view index of -1 is reported, not indexed at SIZE_MAX", "[accessor][malformed]")
{
    auto fixture = floatAccessor({1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);

    // glTF allows -1 for an absent index, and an unchecked static_cast<size_t> of it reads a
    // vector at SIZE_MAX. This is the case the guards were added for.
    fixture.accessor.bufferView = -1;

    const auto read = AccessorUtility::get<glm::vec3>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(read.has_value());
    REQUIRE(read.error() == "accessor names buffer view -1 of 1");
}

TEST_CASE("a buffer view index past the end is reported", "[accessor][malformed]")
{
    auto fixture = floatAccessor({1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);
    fixture.accessor.bufferView = 5;

    const auto read = AccessorUtility::get<glm::vec3>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(read.has_value());
    REQUIRE(read.error() == "accessor names buffer view 5 of 1");
}

TEST_CASE("a buffer index of -1 on the view is reported", "[accessor][malformed]")
{
    auto fixture = floatAccessor({1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);
    fixture.model.bufferViews.front().buffer = -1;

    const auto read = AccessorUtility::get<glm::vec3>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(read.has_value());
    REQUIRE(read.error() == "buffer view names buffer -1 of 1");
}

TEST_CASE("a component type wider than the destination is reported before it overruns", "[accessor][malformed]")
{
    auto fixture = floatAccessor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, TINYGLTF_TYPE_VEC3, 1);
    fixture.accessor.componentType = TINYGLTF_COMPONENT_TYPE_DOUBLE;

    // Three doubles is 24 bytes; the destination slot is three floats. The memcpy would have
    // written past the end of every element.
    const auto read = AccessorUtility::get<glm::vec3>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(read.has_value());
    REQUIRE(read.error() == "accessor element is 24 byte(s), wider than the 12 byte(s) the destination "
                            "component type holds");
}

TEST_CASE("an element count past the end of the buffer is reported", "[accessor][malformed]")
{
    auto fixture = floatAccessor({1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);
    fixture.accessor.count = 2;

    const auto read = AccessorUtility::get<std::vector<glm::vec3>>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(read.has_value());
    REQUIRE(read.error() == "accessor claims 2 element(s) of stride 12, past the end of its 12 byte buffer");
}

TEST_CASE("a byte offset past the end of the buffer is reported", "[accessor][malformed]")
{
    auto fixture = floatAccessor({1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);
    fixture.accessor.byteOffset = 64;

    const auto read = AccessorUtility::get<glm::vec3>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(read.has_value());
    REQUIRE(read.error() == "accessor starts at byte 64 of a 12 byte buffer and needs 12 more");
}

TEST_CASE("an accessor holding fewer components than the type needs is reported", "[accessor][malformed]")
{
    const auto fixture = floatAccessor({1.0f, 2.0f, 3.0f}, TINYGLTF_TYPE_VEC3, 1);

    // Reading a vec4 out of a VEC3 accessor used to return vec4(1) — a plausible-looking value
    // with nothing to distinguish it from data the file really held.
    const auto read = AccessorUtility::get<glm::vec4>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(read.has_value());
    REQUIRE(read.error() == "accessor holds 3 component(s), fewer than the 4 this type needs");
}

TEST_CASE("an accessor describing no elements is reported rather than read as a zero", "[accessor][malformed]")
{
    const auto fixture = floatAccessor({1.0f}, TINYGLTF_TYPE_SCALAR, 0);

    const auto scalar = AccessorUtility::get<float>(fixture.model, fixture.accessor);

    REQUIRE_FALSE(scalar.has_value());
    REQUIRE(scalar.error() == "accessor holds 0 component(s), fewer than the 1 this type needs");

    // The vector spelling is the other reading of the same accessor, and there it is not a
    // failure at all: a file may legitimately describe an empty range.
    const auto asVector = AccessorUtility::get<std::vector<float>>(fixture.model, fixture.accessor);

    REQUIRE(asVector.has_value());
    REQUIRE(asVector->empty());
}
