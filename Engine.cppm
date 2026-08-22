module;

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

export module raceengine;

export import raceengine.resource;
export import raceengine.graphics.models;
export import raceengine.shared;
export import raceengine.async;
export import raceengine.game;
export import raceengine.graphics;
export import raceengine.io;
// The device layer. Same rule as the graphics one: the seam and the pure mapping are exported, the
// two concrete backends are implementation partitions, so importing this costs nobody
// <linux/input.h> or <dinput.h>.
export import raceengine.input;
// Re-exported for the reason every module above it is: a game that had to import a second module to
// reach half the engine is a seam this file exists to close. The rule that keeps the Vulkan backend
// out of here does not reach this one — that leak is 51 MB of vulkan.h, VMA, shaderc and GLFW in
// every importer's BMI, and these partitions' global module fragments carry std and glm, both of
// which the graphics closure already brings. Jolt stays where it is: it is reached through
// `extern "C++"` free functions taking fundamental types, so its -mavx2 never touches a BMI.
export import raceengine.physics;
// The sound, and the same shape again: `raceengine.audio` exports what the car is doing and the seam
// that consumes it, and keeps <fmod_studio.hpp> in an implementation partition nobody importing this
// pays for.
export import raceengine.audio;

// Sandbox code names these unqualified in the global namespace; export import alone
// only surfaces raceengine::X, so re-alias them globally.
export using raceengine::awaitAll;
export using raceengine::Behaviour;
export using raceengine::Camera;
export using raceengine::CreateRenderableModelDTO;
export using raceengine::Drawable;
export using raceengine::Entity;
export using raceengine::FboAttachmentType;
export using raceengine::Key;
export using raceengine::Presenter;
export using raceengine::RenderableModel;
export using raceengine::Scene;
export using raceengine::SceneNode;
export using raceengine::ShaderDescriptor;
export using raceengine::ToneCurve;

namespace raceengine
{

export class Engine
{
private:
    // Declaration order IS initialization order and reverse-destruction order:
    // each member may only depend on members declared above it.
    //
    // The worker pool inverts that edge: it does not depend on the services below it, it
    // *executes code belonging to* them, so it has to stop and join before they are
    // destroyed — which means being declared after every service whose work it runs.
    // ResourceService takes it by reference before it is constructed, which is fine: the
    // reference is only stored, and no job can be submitted until the Engine is built.
    std::shared_ptr<spdlog::logger> logger;
    // The engine's tally of work a frame was asked for and did not do. It belongs to the
    // composition root rather than to the backend because the backend is not the only thing
    // that skips: the skinning path does too, and there is one answer per Engine, not per
    // process — which is what the function-local `static` latches this replaced could not say.
    FrameDiagnostics frameDiagnostics;
    GLFWWindow glfwWindow;
    MemoryStorageService memoryStorageService;
    GLTFService gltfService;
    ResourceService resourceService;
    BackgroundWorkerService backgroundWorkerService;
    RenderableEntityService renderableEntityService;
    SceneManagerService sceneManagerService;
    // unique_ptr because the concrete backend is an implementation partition this header cannot
    // name, but the declaration position is still load-bearing: the device's teardown destroys a
    // surface built from the window, so this member must keep destroying before glfwWindow.
    //
    // This is the only member declared as the whole device: every service below takes the one
    // seam it uses, so a service cannot reach the frame or the factory it has no business in.
    std::unique_ptr<IRenderBackend> renderer;
    FboService fboService;
    ShaderService shaderService;
    CubeMapService cubeMapService;
    // Places probes and marks them stale. It owns no device — a probe is scene data — so it sits
    // with the other authoring services and not with the backend that captures them.
    LightProbeService lightProbeService;
    PostProcessService postProcessService;
    PresenterService presenterService;
    // The release side of the resource model. It needs both the storage (to remove elements) and
    // the backend (to release what those elements held ids for), which is why it cannot sit with
    // ResourceService up above the renderer — loading needs neither.
    AssetService assetService;
    CameraService cameraService;
    // Builds the reduction chain a camera meters itself from and runs the adaptation over it, so
    // it needs the framebuffers, the post-process passes and the camera they hang on — which is
    // what puts it below all three.
    AutoExposureService autoExposureService;
    // Builds the buffers a camera gathers its own occlusion through: the same three dependencies as
    // the meter above, minus the camera service, because it only ever writes the camera it is
    // handed rather than adding passes to that camera's own chain.
    AmbientOcclusionService ambientOcclusionService;
    // Builds the two chains a camera's highlights spill down and back up. Same dependencies as the
    // meter, and for the same reason: buffers, passes, and the camera the passes hang on.
    BloomService bloomService;
    // Turns a .cube file into a table the presenter can name. It needs storage and nothing else — the
    // upload is the backend's, on the frame the grade is first presented with.
    ColourGradeService colourGradeService;
    // Builds and moves the cascade cameras, so it needs the service that builds cameras and
    // nothing else — the depth targets are cameras like any other.
    ShadowService shadowService;
    SceneService sceneService;
    EntityService entityService;
    // The device layer, and the last thing constructed because it is the first thing torn down: it
    // owns a thread, and a thread member stops and joins before anything it could still be touching
    // has been destroyed. It reaches only the logger and the window, both declared far above it, so
    // the backend it was handed outlives it by exactly one member.
    //
    // A unique_ptr for the reason the renderer is one — the concrete backend is an implementation
    // partition this file cannot name — and declared immediately above the service so that the
    // service is destroyed first.
    std::unique_ptr<IInputBackend> inputBackend;
    InputService inputService;
    // The write side, and it is declared after the read side because it goes through it: one object
    // owns the device's lifetime, so a torque cannot be written into a file descriptor the reader
    // has already decided is gone. It owns a thread of its own, which is why it is the last service
    // here — it stops and joins before the service it holds a reference to is destroyed.
    ForceFeedbackService forceFeedbackService;
    AudioService audioService;
    // Simulation clock and the game's tick subscribers. Neither depends on a service, so
    // they sit last: the callbacks are owned by the engine but written by the game, and
    // being destroyed first means no service they reach into is gone while they still exist.
    // Seeded with one step: the first frame's delta is zero, because the window clock has not
    // ticked yet, and a frame that renders before any tick has run would show the state the
    // level was built with rather than the state its first update produced.
    float accumulator = fixedTimeStep;
    float interpolationAlpha = 0.0f;
    std::vector<std::function<void(float)>> updateCallbacks;
    // How the engine asks its own loop to stop, and what the process should say when it does.
    // The frame capture is the one thing that ends a run from below `main` today; it used to do
    // it with std::exit, which skipped every destructor between here and the process — the
    // renderer's device teardown, the window, the worker pool — and would take a test runner
    // with it.
    bool stopRequested = false;
    int exitStatus = 0;

public:
    // Simulation tick length. 120 Hz: 8.33 ms keeps the stiff spring/damper integration a
    // driving game needs stable, and 120 divides the refresh rates a vsynced frame actually
    // lands on (60/120/240) so a frame consumes a whole number of ticks instead of beating
    // against them. Games integrate against this constant, so it is part of the surface.
    static constexpr float fixedTimeStep = 1.0f / 120.0f;
    // Spiral-of-death guard: the most ticks one frame may run. Beyond it the surplus is
    // discarded rather than deferred, so simulation slows down below ~15 fps instead of the
    // catch-up work making the next frame longer, which makes the next catch-up longer. The
    // frame this actually fires on is the first: delta there is the whole process startup,
    // asset load included, which is thousands of ticks of "owed" time that never happened.
    static constexpr int maxCatchUpSteps = 8;

    Engine();
    // Declared only because the physics backend's process-wide registry has to come down with the
    // engine that stood it up. It makes Engine non-movable, which it already was in practice: a
    // game holds one by value and the services below hold references into each other.
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    // False once the window has been closed or the engine has asked its own loop to stop. A
    // game's loop is `while (engine.running()) engine.step();` and nothing below it ends the
    // process, so every destructor between the loop and `main` still runs.
    [[nodiscard]] bool running() const;
    // What the process should return. Zero unless something below the loop asked to stop and
    // had a failure to report — today that is only the frame capture.
    [[nodiscard]] int status() const
    {
        return exitStatus;
    }

    void step();
    // Game update logic on the fixed tick, run in registration order at the top of a tick —
    // before entity behaviours and before the scene settles, and always before the frame
    // that renders its effects. Registration is add-only, like the rest of the engine.
    void onUpdate(std::function<void(float)> callback);

    // Fraction of a fixed step that was left unsimulated when this frame was drawn, in
    // [0, 1): what a renderer interpolates by to draw between two ticks.
    [[nodiscard]] float interpolation() const
    {
        return interpolationAlpha;
    }

    // The engine's own logger, so a game can say something into the same stream the engine does
    // rather than starting a second one that lands somewhere else in the file. Named `log` because
    // the member it returns is called `logger` and the two may not share a name.
    [[nodiscard]] spdlog::logger& log()
    {
        return *logger;
    }

    [[nodiscard]] IWindow& window()
    {
        return glfwWindow;
    }

    [[nodiscard]] ResourceService& resource()
    {
        return resourceService;
    }

    [[nodiscard]] MemoryStorageService& memoryStorage()
    {
        return memoryStorageService;
    }

    [[nodiscard]] SceneManagerService& sceneManager()
    {
        return sceneManagerService;
    }

    [[nodiscard]] SceneService& scene()
    {
        return sceneService;
    }

    [[nodiscard]] RenderableEntityService& renderableEntity()
    {
        return renderableEntityService;
    }

    [[nodiscard]] CameraService& camera()
    {
        return cameraService;
    }

    [[nodiscard]] ShadowService& shadow()
    {
        return shadowService;
    }

    [[nodiscard]] ShaderService& shader()
    {
        return shaderService;
    }

    [[nodiscard]] CubeMapService& cubeMap()
    {
        return cubeMapService;
    }

    [[nodiscard]] LightProbeService& lightProbe()
    {
        return lightProbeService;
    }

    [[nodiscard]] FboService& fbo()
    {
        return fboService;
    }

    [[nodiscard]] PostProcessService& postProcess()
    {
        return postProcessService;
    }

    [[nodiscard]] AutoExposureService& autoExposure()
    {
        return autoExposureService;
    }

    [[nodiscard]] AmbientOcclusionService& ambientOcclusion()
    {
        return ambientOcclusionService;
    }

    [[nodiscard]] BloomService& bloom()
    {
        return bloomService;
    }

    [[nodiscard]] ColourGradeService& colourGrade()
    {
        return colourGradeService;
    }

    [[nodiscard]] PresenterService& presenter()
    {
        return presenterService;
    }

    [[nodiscard]] AssetService& asset()
    {
        return assetService;
    }

    [[nodiscard]] EntityService& entity()
    {
        return entityService;
    }

    // Sampled once per fixed tick by whoever is driving. A wheel, a pad and the keyboard all come
    // out of here as the same struct, and nothing below this line ever learns which it was.
    [[nodiscard]] InputService& input()
    {
        return inputService;
    }

    // Where a game publishes stage one — steering rack torque in newton metres — once per physics
    // tick. Everything below it is this device's problem and nothing above it may know what device
    // that is.
    // Where a game says what the car is doing, in the units a sound bank names. One call a tick, and
    // nothing below it knows what a vehicle is — the same rule stage one of the force feedback keeps.
    [[nodiscard]] AudioService& audio()
    {
        return audioService;
    }

    [[nodiscard]] ForceFeedbackService& forceFeedback()
    {
        return forceFeedbackService;
    }

private:
    void update(float delta);
    [[nodiscard]] float frameDelta() const;
    void recordProbeCaptures();
    void dumpFrameIfRequested();
};

} // namespace raceengine
