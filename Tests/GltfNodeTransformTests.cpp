#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_gltf.h>

import raceengine.graphics.models;
import raceengine.io;
import raceengine.shared;
import raceengine.tests.log;

using raceengine::GLTFService;
using raceengine::MemoryStorageService;
using raceengine::tests::CapturedLog;

namespace
{

// glTF lets a node state its transform *either* as sixteen numbers or as translation/rotation/
// scale, and an exporter picks one for the whole file: Blender writes TRS, the FBX and Collada
// paths most car models arrive through write matrices. Both forms are exercised here because a
// loader reading only one of them does not fail — it silently uses the identity, and the model
// comes out as its parts piled at the origin.
//
// A mesh with no primitives is enough: processNode records the transform before it walks them, so
// this pins the arithmetic without an accessor, a buffer or a warning about geometry that is not
// what is under test.
tinygltf::Model oneNode()
{
    tinygltf::Model model;

    model.meshes.emplace_back();

    tinygltf::Node node;
    node.mesh = 0;
    model.nodes.push_back(node);

    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    model.scenes.push_back(scene);

    return model;
}

glm::mat4 onlyMeshMatrix(const MemoryStorageService& storage, const raceengine::Model& model)
{
    REQUIRE(model.meshes.size() == 1);
    const auto* mesh = storage.meshes.find(model.meshes.front());
    REQUIRE(mesh != nullptr);

    return mesh->modelMatrix;
}

void requireNear(const glm::mat4& actual, const glm::mat4& expected)
{
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            REQUIRE(std::abs(actual[column][row] - expected[column][row]) < 1e-5f);
        }
    }
}

} // namespace

TEST_CASE("a node's transform is read from its matrix as well as from its TRS", "[gltf][transform]")
{
    const auto expected = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, -3.0f, 7.0f)) *
                          glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f));

    CapturedLog log;
    MemoryStorageService storage;
    const GLTFService gltfService(log.sink(), storage);

    auto asTrs = oneNode();
    asTrs.nodes[0].translation = {2.0, -3.0, 7.0};
    asTrs.nodes[0].scale = {0.5, 0.5, 0.5};

    auto asMatrix = oneNode();
    // Column-major, which is the file's order and glm's, so the two agree element for element.
    asMatrix.nodes[0].matrix.assign(&expected[0][0], &expected[0][0] + 16);

    const auto fromTrs = gltfService.gltfModelToInternal("trs.gltf", asTrs);
    const auto fromMatrix = gltfService.gltfModelToInternal("matrix.gltf", asMatrix);
    REQUIRE(fromTrs.has_value());
    REQUIRE(fromMatrix.has_value());

    requireNear(onlyMeshMatrix(storage, fromTrs.value()), expected);
    requireNear(onlyMeshMatrix(storage, fromMatrix.value()), expected);
}

// The failure this guards against is a child placed by its parent's matrix rather than by the
// composition of both: one level deep it looks like a slightly wrong position, and at the depth a
// car body is exported to it looks like a model with no scale at all.
TEST_CASE("a matrix-stated node composes with its parent", "[gltf][transform]")
{
    const auto parentTransform = glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 0.0f, 0.0f));
    const auto childTransform = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 2.0f, 2.0f));

    CapturedLog log;
    MemoryStorageService storage;
    const GLTFService gltfService(log.sink(), storage);

    tinygltf::Model tinyGltfModel;
    tinyGltfModel.meshes.emplace_back();

    tinygltf::Node parent;
    parent.matrix.assign(&parentTransform[0][0], &parentTransform[0][0] + 16);
    parent.children.push_back(1);
    tinyGltfModel.nodes.push_back(parent);

    tinygltf::Node child;
    child.mesh = 0;
    child.matrix.assign(&childTransform[0][0], &childTransform[0][0] + 16);
    tinyGltfModel.nodes.push_back(child);

    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    tinyGltfModel.scenes.push_back(scene);

    const auto model = gltfService.gltfModelToInternal("nested.gltf", tinyGltfModel);
    REQUIRE(model.has_value());

    requireNear(onlyMeshMatrix(storage, model.value()), parentTransform * childTransform);
}
