module;

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <functional>
#include <memory>
#include <stdexcept>
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
// Re-exported for the reason every module above it is: a game that had to import a second module to
// reach half the engine is a seam this file exists to close. The rule that keeps the Vulkan backend
// out of here does not reach this one — that leak is 51 MB of vulkan.h, VMA, shaderc and GLFW in
// every importer's BMI, and these partitions' global module fragments carry std and glm, both of
// which the graphics closure already brings. Jolt stays where it is: it is reached through
// `extern "C++"` free functions taking fundamental types, so its -mavx2 never touches a BMI.
export import raceengine.physics;

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

private:
    void update(float delta);
    [[nodiscard]] float frameDelta() const;
    void recordProbeCaptures();
    void dumpFrameIfRequested();
};

} // namespace raceengine

module :private;

namespace raceengine
{

Engine::Engine() :
    logger(spdlog::stdout_color_mt<spdlog::async_factory>("engine")),
    frameDiagnostics(*logger),
    glfwWindow(*logger),
    gltfService(*logger, memoryStorageService),
    resourceService(*logger, memoryStorageService, backgroundWorkerService, gltfService),
    renderableEntityService(memoryStorageService, frameDiagnostics),
    renderer(createRenderer(*logger, frameDiagnostics, glfwWindow, glfwWindow, renderableEntityService,
                            sceneManagerService, memoryStorageService)),
    fboService(memoryStorageService, *renderer),
    shaderService(memoryStorageService, *renderer),
    cubeMapService(*renderer, memoryStorageService),
    postProcessService(memoryStorageService, fboService, glfwWindow),
    presenterService(*renderer),
    assetService(*logger, memoryStorageService, *renderer, sceneManagerService),
    cameraService(memoryStorageService, fboService, glfwWindow),
    autoExposureService(*logger, memoryStorageService, fboService, postProcessService, cameraService),
    ambientOcclusionService(*logger, memoryStorageService, fboService, postProcessService),
    bloomService(*logger, memoryStorageService, fboService, postProcessService, cameraService),
    colourGradeService(*logger, memoryStorageService),
    shadowService(cameraService),
    sceneService(renderableEntityService, cameraService, sceneManagerService)
{
    // An engine whose device would not come up has nothing left to do: every service below
    // was built against it, and there is no second backend to fall back to.
    // Thrown, not logged and thrown: main's boundary is what reports an engine that could not be
    // built, and saying it twice through two channels is how a reader ends up looking for two
    // problems.
    if (const auto initialised = renderer->init(); !initialised)
    {
        throw std::runtime_error("Renderer initialisation failed: " + initialised.error());
    }

    renderer->setViewport(glfwWindow.state().windowWidth, glfwWindow.state().windowHeight);

    glfwWindow.onResize(
        [&](int width, int height)
        {
            logger->info("Window Resized: {}px x {}px", width, height);
            renderer->setViewport(width, height);

            for (auto& scenePtr : sceneManagerService.getScenes())
            {
                if (!scenePtr)
                {
                    continue;
                }

                for (auto& camera : scenePtr->cameras)
                {
                    // A camera that owns a target of its own resolution is not following the
                    // window: a 2048x2048 shadow cascade rebuilt at the window's size would lose
                    // the resolution it was asked for, and its framing is not the window's aspect
                    // either. Its post-process chain is its own for the same reason.
                    if (!camera.tracksWindowSize)
                    {
                        continue;
                    }

                    cameraService.setAspectRatio(camera, static_cast<float>(width) / static_cast<float>(height));

                    // This runs inside a GLFW callback, so there is no caller to return the
                    // failure to and no frame boundary to abandon it at: an unrebuilt buffer
                    // is reported and the next resize gets another attempt. Throwing here
                    // would unwind through C frames instead.
                    if (const auto recreated = cameraService.recreateOutputBuffer(camera, width, height); !recreated)
                    {
                        logger->error("Camera output buffer was not rebuilt at {}x{}: {}", width, height,
                                      recreated.error());
                    }

                    for (auto postProcess : camera.postProcesses)
                    {
                        // A pass over a buffer a service sized is that service's to rebuild, and
                        // the exposure meter's is not the window's size at all.
                        const auto* pass = memoryStorageService.postProcesses.find(postProcess);
                        if (pass == nullptr || !pass->tracksWindowSize)
                        {
                            continue;
                        }

                        if (const auto recreated = postProcessService.recreateOutputBuffer(postProcess, width, height);
                            !recreated)
                        {
                            logger->error("Post-process output buffer was not rebuilt at {}x{}: {}", width, height,
                                          recreated.error());
                        }
                    }

                    // The occlusion buffers are not in the chain above and are the view's own size,
                    // which is the whole of why they have to be rebuilt here: the shading pass reads
                    // them at its own pixel, so one left at the old size puts a scaled ghost of the
                    // scene on every surface.
                    if (const auto rebuilt = ambientOcclusionService.resize(camera, width, height); !rebuilt)
                    {
                        logger->error("Ambient occlusion buffers were not rebuilt at {}x{}: {}", width, height,
                                      rebuilt.error());
                    }

                    // Half-size buffers a service owns, so the same rule and the same reason.
                    if (const auto rebuilt = bloomService.resize(camera, width, height); !rebuilt)
                    {
                        logger->error("Bloom chains were not rebuilt at {}x{}: {}", width, height, rebuilt.error());
                    }
                }
            }
        });

    // Last in the body, so nothing above it can throw past a registry that is already up. Jolt's
    // allocator, factory and type registry are process-wide and have to stand up before the first
    // PhysicsWorld and come down after the last, which makes them the composition root's and
    // nobody else's — two owners of a process-wide singleton is the bug it would be hiding. A game
    // holds its worlds in members declared after its engine and is bracketed by that.
    if (const auto physics = bringUpJolt(); !physics)
    {
        throw std::runtime_error(physics.error());
    }
}

Engine::~Engine()
{
    tearDownJolt();
}

bool Engine::running() const
{
    return !stopRequested && !glfwWindow.shouldClose();
}

void Engine::onUpdate(std::function<void(float)> callback)
{
    updateCallbacks.push_back(std::move(callback));
}

// One tick of simulation, always fixedTimeStep long. Order is writers before readers: the
// game's own logic, then the behaviour each entity carries, then the scene settling what
// both of them moved.
void Engine::update(float delta)
{
    for (const auto& callback : updateCallbacks)
    {
        callback(delta);
    }

    entityService.update(delta);

    // A destroyed scene leaves its slot behind so the surviving scenes keep their addresses; the
    // slot is not a scene and every walk skips it.
    for (auto& scenePtr : sceneManagerService.getScenes())
    {
        if (scenePtr)
        {
            sceneService.update(*scenePtr, delta);
        }
    }
}

// The frame's real elapsed time — except under a frame capture, which is a gate that
// compares two backends pixel for pixel. A simulation advanced by however fast each backend
// happened to run would put the two captures at different simulated instants, so a capture
// run advances exactly one tick per frame instead. That also makes a capture reproducible
// across sessions rather than only within one.
float Engine::frameDelta() const
{
    static const bool capturing = std::getenv("RACEENGINE_DUMP_FRAME") != nullptr;

    if (capturing)
    {
        return fixedTimeStep;
    }

    return glfwWindow.delta();
}

void Engine::step()
{
    frameDiagnostics.beginFrame();

    const auto delta = frameDelta();

    // Clamping the accumulator, not the loop, is what bounds catch-up: time beyond the
    // budget is dropped here, so the loop can never find more than maxCatchUpSteps of work
    // and the leftover is always a fraction of one step.
    accumulator = std::min(accumulator + delta, fixedTimeStep * static_cast<float>(maxCatchUpSteps));

    // Counted, because it is the only clock anything temporal below may read. A capture run makes
    // the tick count a function of the frame number rather than of how fast the machine ran, and
    // an exposure adaptation driven by anything else would put a different image on disk.
    auto ticks = 0u;

    while (accumulator >= fixedTimeStep)
    {
        update(fixedTimeStep);
        accumulator -= fixedTimeStep;
        ticks++;
    }

    interpolationAlpha = accumulator / fixedTimeStep;

    for (auto& scenePtr : sceneManagerService.getScenes())
    {
        if (!scenePtr)
        {
            continue;
        }

        // Before the matrices, not after: refitting a cascade *is* choosing the position,
        // direction and orthographic volume its matrix is then built from.
        shadowService.update(*scenePtr);

        for (auto& camera : scenePtr->cameras)
        {
            // Before the frame, not after it: the exposure a view is recorded with is a push
            // constant read while that view's post chain is being recorded, so a camera that
            // adapted afterwards would always be showing the previous frame's number.
            autoExposureService.update(camera, ticks, fixedTimeStep);
            cameraService.updateModelViewProjectionMatrix(camera);
        }
    }

    // The frame is the composition root's, start to finish: it opens once, every camera of
    // every scene records its view into it, the presenter records the pass that reaches the
    // screen, and one endFrame submits and presents the lot. Nothing below opens a frame of
    // its own, which is what makes N cameras N views inside one present rather than N
    // presents of which the first N-1 are empty.
    //
    // A backend that cannot open a frame — a swapchain gone out of date, a minimised window —
    // records nothing this step and skips the close, which is the one path with no present.
    if (renderer->beginFrame())
    {
        // The cascades first: a shadow cascade produces the depth map everything downstream
        // samples, and a producer recorded after its consumer is read before it is written (see
        // CameraRole). Separate passes rather than a sort because the order is fixed groups, and a
        // sort would allocate inside the frame to say so.
        for (auto& scenePtr : sceneManagerService.getScenes())
        {
            if (!scenePtr)
            {
                continue;
            }

            for (auto& camera : scenePtr->cameras)
            {
                if (camera.role == CameraRole::ShadowCascade)
                {
                    renderer->recordView(*scenePtr, camera, delta);
                }
            }
        }

        // Then one probe's worth of capture, between the two: a probe shades from the cascades
        // recorded above, and the scene cameras below shade from the probe.
        recordProbeCaptures();

        for (auto& scenePtr : sceneManagerService.getScenes())
        {
            if (!scenePtr)
            {
                continue;
            }

            for (auto& camera : scenePtr->cameras)
            {
                if (camera.role == CameraRole::Scene)
                {
                    // Immediately before the view that samples it, and inside the same frame: the
                    // occlusion is gathered from this camera's own geometry, so it is neither a
                    // group of its own above nor something a game could order for itself.
                    renderer->recordAmbientOcclusion(*scenePtr, camera, delta);
                    renderer->recordView(*scenePtr, camera, delta);

                    // Immediately after the view that filled it. The reduction this reads from is
                    // the tail of that view's post-process chain, so the copy has to be a command
                    // in the same frame and after those passes; what it copies reaches the CPU a
                    // fixed number of submissions later, never this one.
                    renderer->recordAutoExposure(camera);
                }
            }
        }

        presenterService.record();

        renderer->endFrame();
    }

    // The frame owns the report as it owns the frame: every recorder and the skinning path have
    // finished counting by here, and a reason is stated once rather than at each site that met it.
    frameDiagnostics.report();

    // After the present and before the swap: the capture reads what this frame put on screen.
    dumpFrameIfRequested();

    glfwWindow.swapBuffers();
}

// One probe's worth of capture per frame, over every scene.
//
// Deliberately one, and deliberately in scene-then-probe order rather than by any measure of
// urgency: a capture is six full scene passes plus a prefilter, so the budget is what bounds the
// cost, and a fixed order is what makes the schedule a function of the frame number. That matters
// beyond tidiness — the frame gate captures frame 120, and a probe that reached its sixth face on
// a different frame because the machine was busier would put a different image on disk.
//
// A probe already Ready is skipped, so a settled scene does no capture work at all and a
// time-of-day change costs six frames per probe until the scene has settled again.
void Engine::recordProbeCaptures()
{
    for (auto& scenePtr : sceneManagerService.getScenes())
    {
        if (!scenePtr)
        {
            continue;
        }

        for (auto& probe : scenePtr->probes)
        {
            if (probe.state == LightProbeState::Ready)
            {
                continue;
            }

            renderer->recordProbeCapture(*scenePtr, probe);

            return;
        }
    }
}

// The frame gate's exit code is the whole result: it says a PNG of frame 120 exists on disk.
// A capture that reported a failure has written nothing, so returning 0 would hand the gate a
// stale file — or no file — and call it a pass.
//
// The capture ends the run by asking the loop to stop, not by ending the process. std::exit here
// skipped every automatic destructor between this call and main — the Vulkan device, the GLFW
// window, the worker pool — which is why the recorded LSan noise on this path was third-party
// allocations GLFW would have released in glfwTerminate. It would also have killed a test runner
// stone dead, and a test suite is what comes next.
void Engine::dumpFrameIfRequested()
{
    static const char* dumpPath = std::getenv("RACEENGINE_DUMP_FRAME");
    if (dumpPath == nullptr)
    {
        return;
    }

    static int dumpFrameCount = 0;
    if (++dumpFrameCount < 120)
    {
        return;
    }

    stopRequested = true;

    if (const auto captured = renderer->captureFrame(dumpPath); !captured)
    {
        logger->error("Frame capture to {} did not produce a file: {}", dumpPath, captured.error());
        exitStatus = 1;
    }
}

} // namespace raceengine
