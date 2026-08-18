#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using Catch::Approx;
using raceengine::SceneManagerService;

TEST_CASE("createScene reuses the slot a destroyed scene left behind", "[scene]")
{
    SceneManagerService sceneManager;

    auto& first = sceneManager.createScene();
    auto& second = sceneManager.createScene();
    auto* secondAddress = &second;

    REQUIRE(&first != &second);
    REQUIRE(sceneManager.getScenes().size() == 2);

    sceneManager.destroyScene(first);

    // The slot survives so that everything reached through the *other* scene keeps its address:
    // SceneNode&, RenderableEntity::node and a game's own pointer into scene.models are all raw
    // references into that scene's deques.
    REQUIRE(sceneManager.getScenes().size() == 2);
    REQUIRE(&sceneManager.getScene(1) == secondAddress);
    REQUIRE_THROWS_AS(sceneManager.getScene(0), std::out_of_range);

    // And the next scene lands in it rather than growing the deque, so a level cycle is bounded.
    auto& third = sceneManager.createScene();

    REQUIRE(sceneManager.getScenes().size() == 2);
    REQUIRE(&sceneManager.getScene(0) == &third);
}

TEST_CASE("an unparented node's world transform is its own transform", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    sceneManager.setPosition(node, 3.0f, 4.0f, 5.0f);
    sceneManager.setScale(node, 2.0f, 2.0f, 2.0f);

    const auto& world = sceneManager.modelMatrix(node);

    REQUIRE(world[3].x == Approx(3.0));
    REQUIRE(world[3].y == Approx(4.0));
    REQUIRE(world[3].z == Approx(5.0));
    REQUIRE(world[0].x == Approx(2.0));
}

TEST_CASE("translate accumulates on top of what is already set", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    sceneManager.setPosition(node, 1.0f, 0.0f, 0.0f);
    sceneManager.translate(node, 2.0f, 3.0f, 0.0f);

    // The stored position and the matrix agree here, which is what makes the two cases below
    // stand out rather than look like the house style.
    REQUIRE(node.position.x == Approx(3.0));
    REQUIRE(sceneManager.modelMatrix(node)[3].x == Approx(3.0));
    REQUIRE(sceneManager.modelMatrix(node)[3].y == Approx(3.0));
}

// Characterization of a second unit disagreement, of the same family as the rotate/setDirection
// one CLAUDE.md records: scale() *adds* to the stored scale vector and *multiplies* into the
// scale matrix, so node.scale and the matrix it is supposed to describe part company after the
// first call. Nothing reads node.scale today, which is why it has never shown up.
TEST_CASE("scale adds to the stored vector and multiplies into the matrix", "[scene][transform][known-issue]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    sceneManager.setScale(node, 2.0f, 2.0f, 2.0f);
    sceneManager.scale(node, 3.0f, 3.0f, 3.0f);

    REQUIRE(node.scale.x == Approx(5.0));
    REQUIRE(sceneManager.modelMatrix(node)[0].x == Approx(6.0));
}

TEST_CASE("a child's world transform is its parent's applied to its own", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& parent = sceneManager.createNode(scene);
    auto& child = sceneManager.createNode(scene);

    sceneManager.setPosition(parent, 10.0f, 0.0f, 0.0f);
    sceneManager.setPosition(child, 0.0f, 5.0f, 0.0f);
    sceneManager.setParent(child, parent);

    const auto& world = sceneManager.modelMatrix(child);

    REQUIRE(world[3].x == Approx(10.0));
    REQUIRE(world[3].y == Approx(5.0));
}

TEST_CASE("a parent's scale reaches the child's world position", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& parent = sceneManager.createNode(scene);
    auto& child = sceneManager.createNode(scene);

    sceneManager.setScale(parent, 3.0f, 3.0f, 3.0f);
    sceneManager.setPosition(child, 0.0f, 5.0f, 0.0f);
    sceneManager.setParent(child, parent);

    // Composition order is parent * (T * R * S), so the parent's scale multiplies the child's
    // offset rather than only its size.
    REQUIRE(sceneManager.modelMatrix(child)[3].y == Approx(15.0));
}

TEST_CASE("moving a parent invalidates the child's cached world transform", "[scene][transform][cache]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& parent = sceneManager.createNode(scene);
    auto& child = sceneManager.createNode(scene);

    sceneManager.setParent(child, parent);
    sceneManager.setPosition(parent, 10.0f, 0.0f, 0.0f);
    sceneManager.setPosition(child, 0.0f, 5.0f, 0.0f);

    REQUIRE(sceneManager.modelMatrix(child)[3].x == Approx(10.0));

    // Nothing about the child changed, so its own dirty flag is still false. The only thing that
    // can force the recompute is the parent's transformVersion no longer matching the version
    // this cache was built against — which is the whole reason those two counters exist.
    sceneManager.setPosition(parent, 20.0f, 0.0f, 0.0f);

    REQUIRE(sceneManager.modelMatrix(child)[3].x == Approx(20.0));
    REQUIRE(sceneManager.modelMatrix(child)[3].y == Approx(5.0));
}

TEST_CASE("invalidation reaches a grandchild through the chain", "[scene][transform][cache]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& grandparent = sceneManager.createNode(scene);
    auto& parent = sceneManager.createNode(scene);
    auto& child = sceneManager.createNode(scene);

    sceneManager.setParent(parent, grandparent);
    sceneManager.setParent(child, parent);

    sceneManager.setPosition(grandparent, 1.0f, 0.0f, 0.0f);
    sceneManager.setPosition(parent, 0.0f, 2.0f, 0.0f);
    sceneManager.setPosition(child, 0.0f, 0.0f, 3.0f);

    REQUIRE(sceneManager.modelMatrix(child)[3].x == Approx(1.0));
    REQUIRE(sceneManager.modelMatrix(child)[3].y == Approx(2.0));
    REQUIRE(sceneManager.modelMatrix(child)[3].z == Approx(3.0));

    // Two levels up: the grandparent's recompute bumps the parent's version, which bumps the
    // child's. A single-level invalidation check would leave the child on a stale matrix.
    sceneManager.setPosition(grandparent, 100.0f, 0.0f, 0.0f);

    REQUIRE(sceneManager.modelMatrix(child)[3].x == Approx(100.0));
    REQUIRE(sceneManager.modelMatrix(child)[3].y == Approx(2.0));
    REQUIRE(sceneManager.modelMatrix(child)[3].z == Approx(3.0));
}

TEST_CASE("an unchanged chain is not recomputed", "[scene][transform][cache]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& parent = sceneManager.createNode(scene);
    auto& child = sceneManager.createNode(scene);

    sceneManager.setParent(child, parent);
    sceneManager.setPosition(parent, 1.0f, 0.0f, 0.0f);

    static_cast<void>(sceneManager.modelMatrix(child));
    const auto parentVersionAfterFirst = parent.transformVersion;
    const auto childVersionAfterFirst = child.transformVersion;

    REQUIRE(parentVersionAfterFirst == 1u);
    REQUIRE(childVersionAfterFirst == 1u);

    // The cache is the point: the draw path asks for every node's matrix every frame, and a
    // scene that has not moved must cost a comparison rather than a matrix chain.
    for (auto repeat = 0; repeat < 5; repeat++)
    {
        static_cast<void>(sceneManager.modelMatrix(child));
    }

    REQUIRE(parent.transformVersion == parentVersionAfterFirst);
    REQUIRE(child.transformVersion == childVersionAfterFirst);

    sceneManager.setPosition(child, 0.0f, 1.0f, 0.0f);
    static_cast<void>(sceneManager.modelMatrix(child));

    REQUIRE(child.transformVersion == childVersionAfterFirst + 1u);
    REQUIRE(parent.transformVersion == parentVersionAfterFirst);
}

TEST_CASE("setParent alone dirties the child", "[scene][transform][cache]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& parent = sceneManager.createNode(scene);
    auto& child = sceneManager.createNode(scene);

    sceneManager.setPosition(parent, 7.0f, 0.0f, 0.0f);
    sceneManager.setPosition(child, 0.0f, 0.0f, 0.0f);

    REQUIRE(sceneManager.modelMatrix(child)[3].x == Approx(0.0));

    sceneManager.setParent(child, parent);

    REQUIRE(sceneManager.modelMatrix(child)[3].x == Approx(7.0));
}

// Characterization, not endorsement: CLAUDE.md records that SceneManagerService disagrees with
// itself about angle units — rotate() converts its argument with glm::radians and setDirection()
// uses it raw, and both write the same rotationMatrix. Pinning it here means whoever settles the
// disagreement is told by a failing test rather than by a rendered frame.
TEST_CASE("rotate takes degrees where setDirection takes radians", "[scene][transform][known-issue]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& viaRotate = sceneManager.createNode(scene);
    auto& viaDirection = sceneManager.createNode(scene);

    sceneManager.rotate(viaRotate, 90.0f, 0.0f, 1.0f, 0.0f);
    sceneManager.setDirection(viaDirection, glm::radians(90.0f), 0.0f, 1.0f, 0.0f);

    const auto& fromRotate = sceneManager.modelMatrix(viaRotate);
    const auto& fromDirection = sceneManager.modelMatrix(viaDirection);

    // A quarter turn about +Y sends +X to -Z, and both spellings land there only because one of
    // them converted and the other did not.
    REQUIRE(fromRotate[0].z == Approx(-1.0).margin(1e-6));
    REQUIRE(fromDirection[0].z == Approx(-1.0).margin(1e-6));
}
