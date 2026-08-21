module;

#include <deque>
#include <memory>
#include <stdexcept>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.graphics:SceneManagerService;

import raceengine.graphics.models;

namespace raceengine
{

// Scenes are held indirectly so that destroying one leaves the others where they were: a Scene&
// a game is holding, and every SceneNode& and RenderableModel* inside every *other* scene, has to
// survive its neighbour going away. A destroyed scene leaves an empty slot, which the next
// createScene reuses, so a level cycle does not grow the deque.
export class SceneManagerService
{
private:
    std::deque<std::unique_ptr<Scene>> scenes;

public:
    [[nodiscard]] std::deque<std::unique_ptr<Scene>>& getScenes();
    [[nodiscard]] Scene& getScene(int index);
    [[nodiscard]] Scene& createScene();
    // Destroys the scene and everything it owns: its cameras, lights, renderables and nodes. Every
    // reference into it dies here, which is the whole point of the scene being the unit — see the
    // note on Scene. Whatever the scene's cameras owned on the GPU is AssetService::unloadScene's
    // job, and it calls this last.
    void destroyScene(Scene& scene);
    [[nodiscard]] SceneNode& createNode(Scene& scene);
    [[nodiscard]] const glm::mat4& modelMatrix(SceneNode& node) const;
    void setPosition(SceneNode& node, float x, float y, float z) const;
    void setDirection(SceneNode& node, float angle, float x, float y, float z) const;
    // setDirection's rotation, for a caller that already holds one as a quaternion — a rigid body's
    // attitude, a glTF node's TRS — rather than as an axis and an angle. Single precision because
    // the whole scene graph is: a physics body's glm::dquat narrows at this call, explicitly.
    void setOrientation(SceneNode& node, const glm::quat& orientation) const;
    void setScale(SceneNode& node, float x, float y, float z) const;
    void translate(SceneNode& node, float x, float y, float z) const;
    void rotate(SceneNode& node, float angle, float x, float y, float z) const;
    void scale(SceneNode& node, float x, float y, float z) const;
    void setParent(SceneNode& node, SceneNode& parent) const;
};

std::deque<std::unique_ptr<Scene>>& SceneManagerService::getScenes()
{
    return scenes;
}

// A slot a destroyed scene left behind is not a scene, and asking for it is a caller bug rather
// than a runtime condition — the same reading getScene has always had for an index nothing was
// created at.
Scene& SceneManagerService::getScene(int index)
{
    auto& scene = scenes.at(static_cast<size_t>(index));
    if (!scene)
    {
        throw std::out_of_range("Scene " + std::to_string(index) + " was destroyed");
    }

    return *scene;
}

Scene& SceneManagerService::createScene()
{
    for (auto& slot : scenes)
    {
        if (!slot)
        {
            slot = std::make_unique<Scene>();
            return *slot;
        }
    }

    return *scenes.emplace_back(std::make_unique<Scene>());
}

void SceneManagerService::destroyScene(Scene& scene)
{
    for (auto& slot : scenes)
    {
        if (slot.get() == &scene)
        {
            slot.reset();
            return;
        }
    }
}

SceneNode& SceneManagerService::createNode(Scene& scene)
{
    auto& e = scene.nodes.emplace_back();
    return e;
}

void SceneManagerService::setPosition(SceneNode& node, float x, float y, float z) const
{
    node.position = glm::vec3(x, y, z);
    node.translationMatrix = glm::translate(glm::mat4(1.0f), node.position);
    node.transformDirty = true;
}

void SceneManagerService::setDirection(SceneNode& node, float angle, float x, float y, float z) const
{
    node.rotation = glm::vec4(x, y, z, angle);
    node.rotationMatrix = glm::rotate(glm::mat4(1.0), angle, glm::vec3(x, y, z));
    node.transformDirty = true;
}

// Decomposed and handed to setDirection rather than written as glm::mat4_cast, so that node.rotation
// keeps exactly the reading setDirection gives it — axis in xyz, angle in w, radians. modelMatrix
// reads only rotationMatrix, so no reader of the vec4 can tell which setter wrote a node, and the
// pair cannot drift while one function writes both.
//
// Normalising is not defensive. The decomposition below reads the half-angle off a unit quaternion,
// so an integrator's drifted attitude would come out at a plainly wrong angle rather than a
// slightly wrong one.
//
// glm::axis is deliberately not what recovers the axis: it computes the imaginary part's length as
// sqrt(1 - w*w), which cancels to exactly zero in float for every rotation under about a fiftieth
// of a degree and substitutes +Z there, so a node driven from a body sitting all but level turns
// about an axis nothing asked for. Measuring that length directly has no such band, and the
// fallback below is then reached only by an exactly identity quaternion, which has no axis to lose.
void SceneManagerService::setOrientation(SceneNode& node, const glm::quat& orientation) const
{
    const auto unit = glm::normalize(orientation);
    const auto imaginary = glm::vec3(unit.x, unit.y, unit.z);
    const auto halfAngleSine = glm::length(imaginary);
    const auto rotationAxis = halfAngleSine > 0.0f ? imaginary / halfAngleSine : glm::vec3(0.0f, 0.0f, 1.0f);

    setDirection(node, 2.0f * glm::atan(halfAngleSine, unit.w), rotationAxis.x, rotationAxis.y, rotationAxis.z);
}

void SceneManagerService::setScale(SceneNode& node, float x, float y, float z) const
{
    node.scale = glm::vec3(x, y, z);
    node.scaleMatrix = glm::scale(glm::mat4(1.0f), node.scale);
    node.transformDirty = true;
}

void SceneManagerService::translate(SceneNode& node, float x, float y, float z) const
{
    node.position += glm::vec3(x, y, z);
    node.translationMatrix = glm::translate(node.translationMatrix, glm::vec3(x, y, z));
    node.transformDirty = true;
}

void SceneManagerService::rotate(SceneNode& node, float angle, float x, float y, float z) const
{
    node.rotation += glm::vec4(x, y, z, angle);
    node.rotationMatrix = glm::rotate(node.rotationMatrix, glm::radians(angle), glm::vec3(x, y, z));
    node.transformDirty = true;
}

void SceneManagerService::scale(SceneNode& node, float x, float y, float z) const
{
    node.scale += glm::vec3(x, y, z);
    node.scaleMatrix = glm::scale(node.scaleMatrix, glm::vec3(x, y, z));
    node.transformDirty = true;
}

void SceneManagerService::setParent(SceneNode& node, SceneNode& parent) const
{
    node.parent = &parent;
    node.transformDirty = true;
}

const glm::mat4& SceneManagerService::modelMatrix(SceneNode& node) const
{
    const glm::mat4* parentMatrix = nullptr;
    auto parentVersion = 0u;

    if (node.parent)
    {
        parentMatrix = &modelMatrix(*node.parent);
        parentVersion = node.parent->transformVersion;
    }

    if (node.transformDirty || parentVersion != node.parentTransformVersion)
    {
        node.modelMatrix = node.translationMatrix * node.rotationMatrix * node.scaleMatrix;

        if (parentMatrix)
        {
            node.modelMatrix = *parentMatrix * node.modelMatrix;
        }

        node.forward = normalize(glm::vec3(glm::inverse(node.modelMatrix)[2]));
        node.transformDirty = false;
        node.parentTransformVersion = parentVersion;
        node.transformVersion++;
    }

    return node.modelMatrix;
}

} // namespace raceengine
