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
    // Resolved before the window: the window's client API and the renderer factory both
    // depend on the selection (member init order is the mechanism).
    GraphicsApi graphicsApi;
    GLFWWindow glfwWindow;
    MemoryStorageService memoryStorageService;
    GLTFService gltfService;
    ResourceService resourceService;
    BackgroundWorkerService backgroundWorkerService;
    RenderableEntityService renderableEntityService;
    SceneManagerService sceneManagerService;
    // unique_ptr for backend selection, but the declaration position is still load-bearing:
    // the GL renderer's destructor issues GL deletes that need the GLFW context current, so
    // this member must keep destroying before glfwWindow — same slot the value member had.
    //
    // This is the only member declared as the whole device: every service below takes the one
    // seam it uses, so a service cannot reach the frame or the factory it has no business in.
    std::unique_ptr<IRenderBackend> renderer;
    FboService fboService;
    ShaderService shaderService;
    CubeMapService cubeMapService;
    PostProcessService postProcessService;
    PresenterService presenterService;
    // The release side of the resource model. It needs both the storage (to remove elements) and
    // the backend (to release what those elements held ids for), which is why it cannot sit with
    // ResourceService up above the renderer — loading needs neither.
    AssetService assetService;
    CameraService cameraService;
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
    [[nodiscard]] bool running() const;
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

    [[nodiscard]] ShaderService& shader()
    {
        return shaderService;
    }

    [[nodiscard]] CubeMapService& cubeMap()
    {
        return cubeMapService;
    }

    [[nodiscard]] FboService& fbo()
    {
        return fboService;
    }

    [[nodiscard]] PostProcessService& postProcess()
    {
        return postProcessService;
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
    void dumpFrameIfRequested();
};

} // namespace raceengine

module :private;

namespace raceengine
{

namespace
{

// Renderer backend factory for the composition root; the api was already resolved by
// selectGraphicsApi before the window existed. Both backends take the same services: the
// storage service to write GPU ids back through their Resources, and the scene services
// the draw path reads node transforms and joint matrices from.
[[nodiscard]] std::unique_ptr<IRenderBackend> createRenderer(GraphicsApi graphicsApi, spdlog::logger& logger,
                                                             IWindow& window,
                                                             RenderableEntityService& renderableEntityService,
                                                             SceneManagerService& sceneManagerService,
                                                             MemoryStorageService& memoryStorageService)
{
    if (graphicsApi == GraphicsApi::Vulkan)
    {
        return std::make_unique<VulkanRenderer>(logger, window, renderableEntityService, sceneManagerService,
                                                memoryStorageService);
    }

    return std::make_unique<OpenGLRenderer>(logger, renderableEntityService, sceneManagerService, memoryStorageService);
}

} // namespace

Engine::Engine() :
    logger(spdlog::stdout_color_mt<spdlog::async_factory>("engine")),
    graphicsApi(selectGraphicsApi(*logger)),
    glfwWindow(*logger, graphicsApi),
    gltfService(*logger, memoryStorageService),
    resourceService(*logger, memoryStorageService, backgroundWorkerService, gltfService),
    renderableEntityService(memoryStorageService),
    renderer(createRenderer(graphicsApi, *logger, glfwWindow, renderableEntityService, sceneManagerService,
                            memoryStorageService)),
    fboService(memoryStorageService, *renderer),
    shaderService(memoryStorageService, *renderer),
    cubeMapService(*renderer, memoryStorageService),
    postProcessService(memoryStorageService, fboService, glfwWindow),
    presenterService(*renderer),
    assetService(*logger, memoryStorageService, *renderer, sceneManagerService),
    cameraService(memoryStorageService, fboService, glfwWindow),
    sceneService(renderableEntityService, cameraService, sceneManagerService)
{
    // An engine whose device would not come up has nothing left to do: every service below
    // was built against it, and there is no second backend to fall back to.
    if (const auto initialised = renderer->init(); !initialised)
    {
        logger->error("Renderer initialisation failed: {}", initialised.error());
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
                        if (const auto recreated = postProcessService.recreateOutputBuffer(postProcess, width, height);
                            !recreated)
                        {
                            logger->error("Post-process output buffer was not rebuilt at {}x{}: {}", width, height,
                                          recreated.error());
                        }
                    }
                }
            }
        });
}

bool Engine::running() const
{
    return !glfwWindow.shouldClose();
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
    const auto delta = frameDelta();

    // Clamping the accumulator, not the loop, is what bounds catch-up: time beyond the
    // budget is dropped here, so the loop can never find more than maxCatchUpSteps of work
    // and the leftover is always a fraction of one step.
    accumulator = std::min(accumulator + delta, fixedTimeStep * static_cast<float>(maxCatchUpSteps));

    while (accumulator >= fixedTimeStep)
    {
        update(fixedTimeStep);
        accumulator -= fixedTimeStep;
    }

    interpolationAlpha = accumulator / fixedTimeStep;

    for (auto& scenePtr : sceneManagerService.getScenes())
    {
        if (!scenePtr)
        {
            continue;
        }

        for (auto& camera : scenePtr->cameras)
        {
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
        for (auto& scenePtr : sceneManagerService.getScenes())
        {
            if (!scenePtr)
            {
                continue;
            }

            for (auto& camera : scenePtr->cameras)
            {
                renderer->recordView(*scenePtr, camera, delta);
            }
        }

        presenterService.record();

        renderer->endFrame();
    }

    // After the present and before the swap: the capture reads what this frame put on screen,
    // and on GL the back buffer's contents are undefined once it has been swapped.
    dumpFrameIfRequested();

    glfwWindow.swapBuffers();
}

// The frame gate's exit code is the whole result: it says a PNG of frame 120 exists on disk.
// A capture that reported a failure has written nothing, so exiting 0 would hand the gate a
// stale file — or no file — and call it a pass.
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

    if (const auto captured = renderer->captureFrame(dumpPath); !captured)
    {
        logger->error("Frame capture to {} failed: {}", dumpPath, captured.error());
        std::exit(1);
    }

    std::exit(0);
}

} // namespace raceengine
