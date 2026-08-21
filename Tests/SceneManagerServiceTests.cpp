#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using Catch::Approx;
using raceengine::SceneManagerService;

namespace
{

void requireMatricesAgree(const glm::mat4& actual, const glm::mat4& expected)
{
    for (auto column = 0; column < 4; column++)
    {
        for (auto row = 0; row < 4; row++)
        {
            REQUIRE(actual[column][row] == Approx(static_cast<double>(expected[column][row])).margin(1e-6));
        }
    }
}

} // namespace

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

TEST_CASE("setOrientation with the identity quaternion leaves the node unrotated", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    sceneManager.setPosition(node, 3.0f, 4.0f, 5.0f);
    sceneManager.setOrientation(node, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));

    // The identity is the degenerate case of the decomposition: it has no axis to recover, so the
    // angle is the thing that has to come back zero.
    REQUIRE(node.rotation.w == Approx(0.0).margin(1e-6));

    auto expected = glm::mat4(1.0f);
    expected[3] = glm::vec4(3.0f, 4.0f, 5.0f, 1.0f);

    requireMatricesAgree(sceneManager.modelMatrix(node), expected);
}

TEST_CASE("setOrientation replaces the orientation rather than accumulating", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    const auto quarterTurn = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    sceneManager.setOrientation(node, quarterTurn);
    sceneManager.setOrientation(node, quarterTurn);

    // Twice is once. A setter that composed the way rotate() does would be at a half turn here,
    // which is the whole difference between this and the incremental family.
    REQUIRE(sceneManager.modelMatrix(node)[0].z == Approx(-1.0).margin(1e-6));

    sceneManager.setOrientation(node, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));

    requireMatricesAgree(sceneManager.modelMatrix(node), glm::mat4(1.0f));
}

TEST_CASE("setOrientation writes the world matrix the quaternion states", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    sceneManager.setOrientation(node, glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)));

    // Spelled out rather than built with glm::rotate, which is what the setter reaches for itself.
    // A quarter turn about +Y sends +X to -Z and +Z to +X, and a column is where a basis vector
    // lands, so those two columns are the whole rotation.
    auto expected = glm::mat4(1.0f);
    expected[0] = glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);
    expected[2] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);

    requireMatricesAgree(sceneManager.modelMatrix(node), expected);
}

TEST_CASE("setOrientation and setDirection agree on the same rotation", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& viaQuaternion = sceneManager.createNode(scene);
    auto& viaAxisAngle = sceneManager.createNode(scene);

    const auto axis = glm::normalize(glm::vec3(1.0f, 2.0f, -3.0f));
    const auto angle = glm::radians(37.0f);

    sceneManager.setOrientation(viaQuaternion, glm::angleAxis(angle, axis));
    sceneManager.setDirection(viaAxisAngle, angle, axis.x, axis.y, axis.z);

    requireMatricesAgree(sceneManager.modelMatrix(viaQuaternion), sceneManager.modelMatrix(viaAxisAngle));

    // And the vec4 carries setDirection's reading whichever of the two wrote it — axis in xyz,
    // angle in w, radians. modelMatrix reads only the matrix, so nothing downstream can tell the
    // setters apart and the field has to mean one thing.
    REQUIRE(viaQuaternion.rotation.x == Approx(static_cast<double>(axis.x)).margin(1e-6));
    REQUIRE(viaQuaternion.rotation.y == Approx(static_cast<double>(axis.y)).margin(1e-6));
    REQUIRE(viaQuaternion.rotation.z == Approx(static_cast<double>(axis.z)).margin(1e-6));
    REQUIRE(viaQuaternion.rotation.w == Approx(static_cast<double>(angle)).margin(1e-6));
}

TEST_CASE("setOrientation composes with a parent and invalidates the child", "[scene][transform][cache]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& parent = sceneManager.createNode(scene);
    auto& child = sceneManager.createNode(scene);

    sceneManager.setPosition(child, 0.0f, 0.0f, 5.0f);
    sceneManager.setParent(child, parent);

    REQUIRE(sceneManager.modelMatrix(child)[3].z == Approx(5.0));

    // A quarter turn about +Y carries the child's +Z offset round onto +X, and the child has to
    // find out through the same version counter every other setter bumps.
    sceneManager.setOrientation(parent, glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)));

    REQUIRE(sceneManager.modelMatrix(child)[3].x == Approx(5.0).margin(1e-6));
    REQUIRE(sceneManager.modelMatrix(child)[3].z == Approx(0.0).margin(1e-6));
}

TEST_CASE("setOrientation composes with translation and scale", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    // Deliberately not in composition order: each setter owns one matrix and modelMatrix is what
    // orders them, so the three have to be writable in any order at all.
    sceneManager.setScale(node, 2.0f, 2.0f, 2.0f);
    sceneManager.setOrientation(node, glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)));
    sceneManager.setPosition(node, 1.0f, 2.0f, 3.0f);

    const auto& world = sceneManager.modelMatrix(node);

    // Translation * rotation * scale: the scale is inside the rotation, so the rotated basis comes
    // out two long and the translation is not multiplied by it.
    REQUIRE(world[3].x == Approx(1.0));
    REQUIRE(world[3].y == Approx(2.0));
    REQUIRE(world[3].z == Approx(3.0));
    REQUIRE(world[0].z == Approx(-2.0).margin(1e-6));
    REQUIRE(world[1].y == Approx(2.0).margin(1e-6));
    REQUIRE(world[2].x == Approx(2.0).margin(1e-6));
}

TEST_CASE("setOrientation normalises the quaternion it is given", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    // An integrator's attitude drifts off unit length, and the decomposition recovers the axis
    // length from w rather than measuring it — so five percent of drift is six degrees of rotation
    // here, not five percent of anything.
    const auto drifted = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * 1.05f;

    sceneManager.setOrientation(node, drifted);

    REQUIRE(node.rotation.w == Approx(static_cast<double>(glm::radians(90.0f))).margin(1e-6));
    REQUIRE(sceneManager.modelMatrix(node)[0].z == Approx(-1.0).margin(1e-6));
}

TEST_CASE("setOrientation keeps the axis of a rotation glm::axis cannot resolve", "[scene][transform]")
{
    SceneManagerService sceneManager;
    auto& scene = sceneManager.createScene();
    auto& node = sceneManager.createNode(scene);

    // A thousandth of a degree, which is inside the band where sqrt(1 - w*w) cancels to exactly
    // zero in float: glm::axis answers +Z there, and a node driven from a physics body sitting all
    // but level would be turned about an axis it never asked for. The angle is what says the
    // rotation is negligible; the axis has to survive it.
    sceneManager.setOrientation(node, glm::angleAxis(glm::radians(0.001f), glm::vec3(0.0f, 1.0f, 0.0f)));

    REQUIRE(node.rotation.y == Approx(1.0).margin(1e-6));
    REQUIRE(node.rotation.z == Approx(0.0).margin(1e-6));
    REQUIRE(node.rotation.w == Approx(static_cast<double>(glm::radians(0.001f))).margin(1e-9));
}
