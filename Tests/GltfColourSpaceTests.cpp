#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <tiny_gltf.h>

import raceengine.graphics.models;
import raceengine.io;
import raceengine.shared;
import raceengine.tests.log;

using raceengine::ColourSpace;
using raceengine::GLTFService;
using raceengine::MemoryStorageService;
using raceengine::Resource;
using raceengine::Texture;
using raceengine::tests::CapturedLog;

namespace
{

// A model that is nothing but images, textures and materials: gltfModelToInternal returns early
// on an empty scene list, having built exactly the part under test. Each image is a distinct 1x1
// so the assertions can name one by index and be sure which slot resolved to it.
tinygltf::Model materialWithSlots(const int baseColour, const int metallicRoughness, const int normal,
                                  const int occlusion, const int emissive, const size_t imageCount)
{
    tinygltf::Model model;

    for (size_t i = 0; i < imageCount; i++)
    {
        tinygltf::Image image;
        image.name = "image " + std::to_string(i);
        image.width = 1;
        image.height = 1;
        image.component = 4;
        image.bits = 8;
        image.image = std::vector<unsigned char>{0, 0, 0, 255};
        model.images.push_back(image);

        tinygltf::Texture texture;
        texture.source = static_cast<int>(i);
        model.textures.push_back(texture);
    }

    tinygltf::Material material;
    material.pbrMetallicRoughness.baseColorTexture.index = baseColour;
    material.pbrMetallicRoughness.metallicRoughnessTexture.index = metallicRoughness;
    material.normalTexture.index = normal;
    material.occlusionTexture.index = occlusion;
    material.emissiveTexture.index = emissive;
    model.materials.push_back(material);

    return model;
}

ColourSpace colourSpaceOf(const MemoryStorageService& storage, const Resource<Texture>& handle)
{
    const auto* texture = storage.textures.find(handle);
    REQUIRE(texture != nullptr);

    return texture->colourSpace;
}

} // namespace

TEST_CASE("base colour and emissive load as sRGB and the measurement maps do not", "[gltf][colour]")
{
    CapturedLog log;
    MemoryStorageService storage;
    const GLTFService gltfService(log.sink(), storage);

    const auto model = gltfService.gltfModelToInternal("slots.gltf", materialWithSlots(0, 1, 2, 3, 4, 5));
    REQUIRE(model.has_value());
    REQUIRE(model->materials.size() == 1);

    const auto* material = storage.materials.find(model->materials.front());
    REQUIRE(material != nullptr);
    REQUIRE(material->albedo.has_value());
    REQUIRE(material->emissive.has_value());
    REQUIRE(material->metallicRoughness.has_value());
    REQUIRE(material->normal.has_value());
    REQUIRE(material->occlusion.has_value());

    CHECK(colourSpaceOf(storage, material->albedo.value()) == ColourSpace::Srgb);
    CHECK(colourSpaceOf(storage, material->emissive.value()) == ColourSpace::Srgb);
    CHECK(colourSpaceOf(storage, material->metallicRoughness.value()) == ColourSpace::Linear);
    CHECK(colourSpaceOf(storage, material->normal.value()) == ColourSpace::Linear);
    CHECK(colourSpaceOf(storage, material->occlusion.value()) == ColourSpace::Linear);
}

// The loader shares one Texture per glTF image, so an image reached from both a colour and a
// measurement slot has one colour space to give. It goes to the colour slot: that is the one a
// wrong answer is visible in, and the alternative — a second upload of the same pixels — is a
// reference count, which is a different feature from this one.
TEST_CASE("an image used as both base colour and a measurement map resolves as sRGB", "[gltf][colour]")
{
    CapturedLog log;
    MemoryStorageService storage;
    const GLTFService gltfService(log.sink(), storage);

    const auto model = gltfService.gltfModelToInternal("shared.gltf", materialWithSlots(0, 0, -1, -1, -1, 1));
    REQUIRE(model.has_value());
    REQUIRE(model->materials.size() == 1);

    const auto* material = storage.materials.find(model->materials.front());
    REQUIRE(material != nullptr);
    REQUIRE(material->albedo.has_value());
    REQUIRE(material->metallicRoughness.has_value());
    CHECK(material->albedo.value().index == material->metallicRoughness.value().index);

    CHECK(colourSpaceOf(storage, material->albedo.value()) == ColourSpace::Srgb);
}

// Images are created once for the whole file and materials are walked afterwards, so the answer
// has to be collected across every material before the first Texture exists. A model whose two
// materials disagree about an image is what catches a loader that decided per material instead.
TEST_CASE("colour slots are collected across every material in the file", "[gltf][colour]")
{
    CapturedLog log;
    MemoryStorageService storage;
    const GLTFService gltfService(log.sink(), storage);

    auto tinyGltfModel = materialWithSlots(-1, 0, -1, -1, -1, 2);
    tinygltf::Material second;
    second.pbrMetallicRoughness.baseColorTexture.index = 1;
    tinyGltfModel.materials.push_back(second);

    const auto model = gltfService.gltfModelToInternal("two-materials.gltf", tinyGltfModel);
    REQUIRE(model.has_value());
    REQUIRE(model->materials.size() == 2);

    const auto* measurementOnly = storage.materials.find(model->materials.front());
    const auto* colourOnly = storage.materials.find(model->materials.back());
    REQUIRE(measurementOnly != nullptr);
    REQUIRE(colourOnly != nullptr);
    REQUIRE(measurementOnly->metallicRoughness.has_value());
    REQUIRE(colourOnly->albedo.has_value());

    CHECK(colourSpaceOf(storage, measurementOnly->metallicRoughness.value()) == ColourSpace::Linear);
    CHECK(colourSpaceOf(storage, colourOnly->albedo.value()) == ColourSpace::Srgb);
}
