module;

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/containers/vector.h>
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/maths/soa_transform.h>
#include <ozz/base/maths/vec_float.h>
#include <ozz/options/options.h>

export module raceengine.graphics.models:Scene;

import raceengine.resource;
import :Fbo;
import :LightProbe;
import :Mesh;
import :Shader;
import :Texture;

namespace raceengine
{

export struct SceneNode
{
    SceneNode* parent = nullptr;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::mat4 rotationMatrix = glm::mat4(1.0f);
    glm::mat4 translationMatrix = glm::mat4(1.0f);
    glm::mat4 scaleMatrix = glm::mat4(1.0f);
    glm::vec4 rotation = glm::vec4(0.0f);
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
    glm::vec3 forward = glm::vec3(0.0f);
    // World-transform cache: transformVersion counts recomputes of modelMatrix (0 = never computed);
    // parentTransformVersion is the parent's version this cache was built against (0 = no parent).
    bool transformDirty = true;
    unsigned int transformVersion = 0;
    unsigned int parentTransformVersion = 0;
};

// What the light's geometry is. A point light radiates from `position` and attenuates; a
// directional light has no position that matters and casts along `direction` from infinitely far
// away, which is the light a cascaded shadow map is a shadow map *of*. `direction` is what the
// cascades are fitted against; shading still reads `position` as the direction *towards* the
// light, exactly as it always has, so the two must be opposites of each other for a shadow to land
// where the shading says it should.
export enum class LightType { Point, Directional };

export struct Light
{
    LightType type = LightType::Point;
    glm::vec3 position{};
    glm::vec3 direction{};
    // The four terms both backends upload per light. They are stored as authored rather than
    // derived from color/strength: the renderers hand them to the shaders verbatim, and a
    // product would not reproduce an authored value bit for bit.
    glm::vec3 diffuse{};
    glm::vec3 specular{};
    glm::vec3 ambient{};
    float attenuation = 1.0f;
    glm::vec3 color{};
    float strength{};
};

// How a camera turns view space into clip space. Perspective is the default and the only one any
// camera in a scene has ever used; orthographic is what a directional light's cascade needs,
// because a slice of a directional light's frustum is a box, not a pyramid.
export enum class CameraProjection { Perspective, Orthographic };

// The orthographic view volume's side planes, in view space. Held as four edges rather than a
// width and a height because a cascade's box is fitted to a frustum slice and is not centred on
// the light's axis in general. The near and far planes are Camera's own clipping planes, so an
// orthographic camera and a perspective one state their depth range the same way.
export struct OrthographicVolume
{
    float left = -1.0f;
    float right = 1.0f;
    float bottom = -1.0f;
    float top = 1.0f;
};

// What the frame does with a camera's view. Scene is the ordinary case and the default. A
// ShadowCascade camera *produces* a depth map that Scene cameras sample, so the frame records
// every one of them before any Scene camera whatever order they were appended to the scene in — a
// depth target sampled by a view recorded earlier in the same command buffer would be read before
// it was written, and on Vulkan the layout barrier would be on the wrong side of the read.
//
// ProbeFace is the same producer/consumer relationship one step further out: it draws one face of
// a light probe's environment, which the Scene cameras then shade from. It is not a member of
// Scene::cameras — the backend builds one on the stack per face, because the six views are a fixed
// function of the probe's position and nothing about them is worth a game's authoring.
// DepthNormalPrepass is the third of the same kind: it draws the view a Scene camera is about to
// shade, from that camera's own position, writing surface normals and view depth and nothing else,
// so the pass that gathers ambient occlusion has geometry to gather it from. Like ProbeFace it is
// not a member of Scene::cameras — the backend builds one on the stack from the camera it is a
// prepass for, because every one of its fields except the target and the shader is that camera's,
// and a second Camera a game had to keep in step would be a second answer to where the view is.
export enum class CameraRole { Scene, ShadowCascade, ProbeFace, DepthNormalPrepass };

// The shape of the curve the post chain maps scene radiance through, once the camera's exposure
// has said how much of it there is. All three leave black at black and white at white, so a change
// here moves contrast rather than range, and all three are separable: contrast pivots at middle
// grey, the toe grips the bottom of the output and the shoulder holds the top back.
//
// The defaults are not neutral. A neutral filmic transfer is the safe answer for an engine that
// does not know what it is rendering, and this one does: an outdoor scene with a sun, deep
// building shadow the light probes are meant to keep dark, and a sky with headroom above it. The
// numbers below are the dramatic reading of that — see docs/vulkan-abi.md for what each does to
// the frame.
export struct ToneCurve
{
    // A power law about `toneGreyPivot` in the exposed scene-linear domain, which is a straight
    // line in log-log: it moves nothing at the pivot and moves everything else symmetrically about
    // it. One is the plain filmic transfer.
    float contrast = 1.2f;
    // How hard the curve grips the bottom of its output range. Zero is the plain filmic toe;
    // raising it darkens what is already dark and leaves the highlights where they were.
    float toe = 0.35f;
    // How hard it holds the top back. Zero is the plain filmic shoulder; raising it keeps a bright
    // sky off the clip rather than compressing the whole range to get there.
    float shoulder = 0.1f;
};

// A camera that reads its own exposure off the frame it drew instead of being told one.
//
// The reading is taken from a mip chain the post-process passes reduce the scene colour down to,
// one texel at the end of it, and it reaches the CPU a fixed number of submissions later — so
// everything here is a lag, never a measurement of the frame being recorded. The adaptation that
// closes that gap is advanced by simulation ticks rather than by elapsed time, for the reason the
// light probe scheduler counts frames: the capture gate compares frame 120 exactly, and a camera
// that adapted by however fast the machine happened to run would put a different image on disk.
export struct AutoExposure
{
    bool enabled = false;
    // Exposure compensation in stops, read exactly as the dial on a camera reads: positive opens
    // up, negative holds back. Zero is what the meter itself says, which puts the frame's geometric
    // mean where a reflected-light meter puts middle grey — the neutral answer, and the level's to
    // move. It is a level's decision because the engine renders in relative radiance: how far the
    // picture wants to sit from a photometric meter's reading depends on what a scene decided a sun
    // was worth, which is not something the camera can know.
    float compensation = 0.0f;
    // How much the meter listens to the middle of the frame rather than to all of it, from zero —
    // a flat average over everything, which is what this was — to one, where the corners count for
    // almost nothing. It exists because the flat average is the answer to the wrong question: a
    // player looking up gets a frame that is mostly sky, the mean rises, the meter closes down and
    // the building he is walking past goes black. Every camera ever built weights the centre for
    // exactly this reason, and the subject is in the middle of the frame because that is where the
    // player put it.
    float centreWeighting = 0.0f;
    // How fast the adaptation closes on a new reading, in e-foldings per second of *simulated*
    // time. Two speeds because adaptation is not symmetric: the eye handles a scene getting
    // brighter far faster than one getting darker, which is why walking into sunlight is a moment
    // and walking into a tunnel is not. Both are a great deal quicker here than an eye's, because a
    // camera the player is looking through has to have caught up by the time he has.
    float lightAdaptationSpeed = 6.0f;
    float darkAdaptationSpeed = 3.0f;
    // The range the meter may pick a shutter inside. A camera cannot hold the shutter open for
    // arbitrarily long, and that limit is the one place a slow lens or a low film speed can fail
    // to reach the exposure the meter asked for — which is what makes iso and aperture matter to
    // the picture rather than only to the arithmetic. The scene's radiance is relative, not
    // cd/m², so these bracket the range rather than calibrating it.
    float minShutterTime = 1.0f / 8000.0f;
    float maxShutterTime = 1.0f / 4.0f;
    // The reduction chain the reading is copied out of, and which of its levels is the single
    // texel. Written by AutoExposureService when it builds the chain; nothing else names them.
    Resource<FboAttachment> chain{};
    unsigned int chainLevel = 0;
    // The last completed reading, written by the backend when its deferred copy lands, and zero
    // until the first one does — which is what says "hold the exposure the level started with".
    float measuredLuminance = 0.0f;
    // What the adaptation is holding right now, which is the luminance the exposure below was
    // derived from. Seeded from that exposure when metering is enabled, so a level's manual
    // setExposure is where the adaptation starts rather than something it discards.
    float adaptedLuminance = 0.0f;
    // Whether the settled reading has been stated in the log. Once, when it settles: the number a
    // level author wants is what the scene meters at, not what it read on the way there.
    bool settleLogged = false;
};

// Ambient occlusion gathered from the view's own geometry, for the light the view's geometry does
// not let in.
//
// It is a *camera* property rather than a scene one because it is measured in the screen the camera
// renders: the prepass draws this camera's view into a buffer of view-space normals and depths, one
// fullscreen pass gathers the horizon around every pixel of it, and a second smooths the result
// along surfaces rather than across their edges. What comes out is bound to the shading pass beside
// the shadow cascades, and the PBR shader folds it into the material's own occlusion term — so it
// darkens indirect light and leaves the sun alone, which is what an occlusion factor means.
export struct AmbientOcclusion
{
    bool enabled = false;
    // How much of the gathered occlusion is applied, as a power on the result: 1 is the term the
    // integral produced, above it deepens. A multiplier would darken the unoccluded parts of the
    // image too, and the whole value of a visibility term is that it is 1 where nothing is in the
    // way.
    float strength = 1.0f;
    // How far the horizon search reaches, in world units. It is a world distance rather than a
    // pixel radius because occlusion is a property of the geometry and not of how close the camera
    // happens to be standing: a corner of a building should darken by the same amount from across
    // the street as from beside it. The default suits a scene measured in the tens of units.
    float radius = 40.0f;
    // The targets, written by AmbientOcclusionService when it builds them. `prepass` is the view's
    // normals and depth, `occlusion` is what the shading pass samples, and the passes are the two
    // that fill them — held here because the backend records them as this camera's prepass, in the
    // order they are listed.
    std::optional<Resource<Fbo>> prepass{};
    std::optional<Resource<Shader>> prepassShader{};
    Resource<FboAttachment> occlusion{};
    std::vector<Resource<PostProcess>> passes{};
};

// The light a bright thing spills onto everything around it — a lens and an eye both do it, and a
// frame without it reads as though nothing in it is actually bright, because a display cannot be.
//
// Two chains rather than one: the frame is thresholded and halved down through the first, then
// carried back up the second, each level combining the one below it with the matching level of the
// first. The usual formulation blends additively into a single chain, which needs a pass to read a
// level it is writing; two chains say the same thing with the engine's own rule intact, and the
// pass that combines them is the multi-input case the post-process seam already has.
export struct Bloom
{
    bool enabled = false;
    // Where a pixel starts spilling, in *exposed* radiance — the threshold pass multiplies by the
    // camera's exposure first, so a scene that meters itself does not change how much of it blooms.
    float threshold = 1.0f;
    // How gradually it starts, as a width in the same units. A hard threshold makes a moving
    // highlight pop in and out of the effect at its edge; the knee is what turns that into a fade.
    float knee = 0.5f;
    // How much of the result is added back to the frame before the tone curve.
    float intensity = 0.05f;
    // The most any one texel may contribute, in exposed radiance. The sun's disc is four orders of
    // magnitude over the sky and the chain is stored in half floats, so an unclamped threshold pass
    // reaches infinity on the way down and takes the whole frame white with it. It is also the knob
    // that stops one specular pinprick spilling across the screen, which is the same problem at a
    // smaller scale.
    float maximum = 60.0f;
    // How far each upsample reaches, in texels of the level it is reading. Above one the tent
    // overlaps its neighbours and the spill widens without another level.
    float spread = 1.0f;
    // The buffers, and the passes over them. `result` is the top of the upsample chain and the
    // attachment the tone map adds in; the rest is the engine's bookkeeping for a resize.
    std::optional<Resource<Fbo>> downsample{};
    std::optional<Resource<Fbo>> upsample{};
    Resource<FboAttachment> result{};
    std::vector<Resource<PostProcess>> passes{};
};

export struct Camera
{
    unsigned int iso;
    float aspectRatio;
    float aperture;
    // How long the shutter is open, in seconds: the third leg of the exposure triangle and the one
    // the meter solves for. `exposure` below is derived from all three (see PhysicalCamera), so
    // this is the state and that is the result — setExposure back-solves this so the two agree
    // even when a game states the multiplier directly.
    float shutterTime{};
    float exposure{};
    // How the exposed radiance is mapped onto the display. Exposure above is how much light there
    // is; this is what is done with it.
    ToneCurve toneCurve{};
    // Whether the exposure above is metered off the frame rather than stated by the level.
    AutoExposure autoExposure{};
    // Whether this view gathers its own ambient occlusion before it shades.
    AmbientOcclusion ambientOcclusion{};
    // Whether the bright parts of this view spill into the rest of it before the tone curve.
    Bloom bloom{};
    float fieldOfView;
    CameraRole role = CameraRole::Scene;
    // The shader every draw in this view uses instead of the entity's own. A cascade renders
    // depth and nothing else, and depth does not vary by material, so one shader serves the whole
    // pass; a camera without one shades each entity as that entity asked to be shaded.
    std::optional<Resource<Shader>> overrideShader{};
    CameraProjection projection = CameraProjection::Perspective;
    // Read only when `projection` is Orthographic; `fieldOfView` and `aspectRatio` are read only
    // when it is Perspective. Both sets are kept rather than unioned so switching a camera back
    // and forth does not lose the other mode's framing.
    OrthographicVolume orthographicVolume{};
    // Whether a window resize rebuilds this camera's render target at the new size. True for a
    // camera that fills the screen — every camera built by createCamera() with no argument.
    // False for one that owns a target of its own resolution: a 2048x2048 shadow cascade does not
    // become 1920x1080 because the window did.
    bool tracksWindowSize = true;
    float nearClippingPlane{};
    float farClippingPlane{};
    glm::vec3 position{};
    glm::vec3 direction;
    glm::vec3 roll;
    glm::mat4 modelViewProjectionMatrix{};
    glm::mat4 modelViewMatrix{};
    std::optional<Resource<Fbo>> output;
    std::vector<Resource<PostProcess>> postProcesses{};
};

export enum class RenderableEntityType {
    Mesh,
    Skybox,
};

export struct RenderableEntity
{
    RenderableEntityType type;
    SceneNode& node;
    // Called by the backend immediately around this entity's submission, once per view that
    // draws it. The hooks live on the renderable rather than on the game's Drawable component
    // because this module is the one both sides already depend on: a renderer that had to see
    // a Drawable would make raceengine.graphics import raceengine.game and invert the module
    // graph. Drawable holds them as a view, so a game still writes them through its component.
    std::optional<std::function<void()>> beforeDraw;
    std::optional<std::function<void()>> afterDraw;
    // Whether a shadow cascade's depth pass draws this entity. False for anything that is scenery
    // rather than geometry: a skybox drawn into a cascade fills the whole map at the near plane
    // and puts the entire world in shadow, which is the first thing a shadow map does wrong.
    bool castsShadow = true;
    // Whether a light probe's capture draws this entity. True for the world — buildings, ground,
    // the sky — and false for anything that moves, because a probe is captured once and then
    // shaded from for many frames: a car baked into the environment goes on lighting the street
    // from wherever it was parked when the capture ran. It is a separate flag from castsShadow
    // because the sky is the case that differs: it must not cast, and it is the single most
    // important thing a probe records.
    bool staticGeometry = true;

    explicit RenderableEntity(RenderableEntityType type, SceneNode& node) :
        type(type),
        node(node)
    {
    }
};

export struct RenderableMesh
{
    float animationTime{};
    unsigned int currentAnimationIndex{};
    const Resource<Mesh> mesh;
    ozz::vector<ozz::math::SoaTransform> animationLocalSpaceTransforms{};
    ozz::vector<ozz::math::Float4x4> animationModelSpaceTransforms{};
    std::optional<Resource<std::unique_ptr<ozz::animation::Skeleton>>> skeleton;
    std::vector<Resource<std::unique_ptr<ozz::animation::Animation>>> animations{};
    std::unique_ptr<ozz::animation::SamplingJob::Context> animationCache{};
    // ozz joint index -> glTF skin joint index, validated against both when the skeleton is set
    // so the per-frame palette build needs no lookup guards.
    std::map<int, int> jointMap{};
    // Joint palette, rebuilt in place every frame by RenderableEntityService::joints so the draw
    // path does not allocate one vector per mesh per frame.
    std::vector<glm::mat4> jointTransforms{};
    // A per-instance transform in this mesh's own node frame, multiplied in after the glTF chain:
    // the baked node matrix maps local to model, so what is appended here rotates about the node's
    // own origin and axes. Identity for everything that never moves; a steering wheel that turns
    // and a road wheel that spins are what it exists for. Per instance rather than on the shared
    // Mesh, because two cars from one model must not steer each other.
    glm::mat4 localTransform{1.0f};
};

export struct RenderableModel : public RenderableEntity
{
    Resource<Model> model;
    // The shader this instance was created with. Materials live in shared storage, so the
    // choice cannot be written through to them without one instance rewriting another's.
    Resource<Shader> shader;
    std::vector<RenderableMesh> meshes;

    explicit RenderableModel(SceneNode& node, Resource<Model> model, Resource<Shader> shader,
                             std::vector<RenderableMesh> meshes) :
        RenderableEntity(RenderableEntityType::Mesh, node),
        model(model),
        shader(shader),
        meshes(std::move(meshes))
    {
    }
};

// One slice of the shadow-casting light's frustum, and the camera that draws it.
//
// The camera is a raw pointer into `Scene::cameras` for the same reason `RenderableEntity::node`
// is a reference into `Scene::nodes`: the scene's deques are add-only and the scene is the unit of
// teardown, so an address inside one is valid exactly as long as the scene is. No generational
// handle reaches inside a scene.
export struct ShadowCascade
{
    Camera* camera = nullptr;
    // Where this cascade stops, as a distance along the view camera's axis. The fragment shader
    // picks its cascade by comparing against these.
    float splitDistance = 0.0f;
    // World units per shadow-map texel, and normalised depth per world unit along the light axis.
    // The whole bias budget is expressed in these two, so no cascade needs tuning of its own.
    float texelWorldSize = 1.0f;
    float depthPerWorldUnit = 1.0f;
};

// A scene's cascaded shadow map. Empty `cascades` means the scene casts no shadow, which is a
// legitimate scene rather than a failure — both backends then shade with the lit result.
export struct ShadowCascades
{
    // The light the cascades follow and the camera whose frustum they split. Pointers into the
    // scene's own deques, on the same terms as ShadowCascade::camera.
    Light* light = nullptr;
    Camera* viewCamera = nullptr;
    // Which light index the shaders should apply the shadow to — the position of `light` in
    // Scene::lights, which is the order both backends upload them in.
    unsigned int lightIndex = 0;
    unsigned int resolution = 0;
    // How far from the eye the cascades reach. Beyond it a fragment is lit: there is no map to
    // ask, and the alternative — treating "outside the map" as shadowed — paints the far half of
    // the world black.
    float distance = 0.0f;
    // The split scheme's blend, and how far behind a slice a cascade still catches casters. Kept
    // here rather than only in the DTO because the refit runs every frame and needs both.
    float lambda = 0.5f;
    float casterExtent = 0.0f;
    std::vector<ShadowCascade> cascades{};
};

export struct Scene
{
    // std::deque, and add-only *within a scene*: SceneNode::parent, RenderableEntity::node and a
    // game's own pointer into models are raw references into these containers, and no generational
    // handle reaches inside a scene. The scene is therefore the unit of teardown — destroying it
    // invalidates all of them at once, which is a rule a game can hold, where erasing one node out
    // of the middle is not. There is deliberately no API for the latter.
    std::deque<Camera> cameras;
    std::deque<Light> lights;
    std::deque<RenderableModel> models;
    std::deque<SceneNode> nodes;
    // The image-based lighting graph. A deque on the same add-only terms as the rest: the backend
    // keys a probe's captured radiance by the array slice recorded on the probe itself, and the
    // scheduler walks these in order, so an address into this container is valid as long as the
    // scene is.
    std::deque<LightProbe> probes;

    std::optional<Resource<CubeMap>> environment{};
    ShadowCascades shadows{};
};

} // namespace raceengine
