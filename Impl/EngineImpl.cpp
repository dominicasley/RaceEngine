// Engine bodies. Declarations are in Engine.cppm.
//
// A **module implementation unit** — `module raceengine;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. This is what this file's
// `module :private;` fragment used to be, and it is here because that fragment did not do the job:
// measured, editing inside one still changed the BMI and still rebuilt every importer.
// Measurements and the rule: docs/build-times.md.
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

#include <Profiling/RaceEngineProfile.hpp>

module raceengine;

import raceengine.resource;
import raceengine.graphics.models;
import raceengine.shared;
import raceengine.async;
import raceengine.game;
import raceengine.graphics;
import raceengine.io;
import raceengine.input;
import raceengine.physics;
import raceengine.audio;

namespace raceengine
{

namespace
{

// What this run is allowed to do with the machine's input devices, and the same test every other
// input path here makes: an unattended run owns no hands at the controls, so it opens no device,
// starts no thread and writes no calibration into somebody's home directory. That is what keeps a
// capture byte-identical with and without any of this.
[[nodiscard]] InputOptions engineInputOptions()
{
    auto options = InputOptions{};

    options.unattended =
        std::getenv("RACEENGINE_UNATTENDED") != nullptr || std::getenv("RACEENGINE_DUMP_FRAME") != nullptr;
    options.profileDirectory = options.unattended ? std::string() : defaultProfileDirectory();

    // The three the two platforms genuinely differ on, stated here rather than discovered as a
    // force that never arrives. Constant force is what the force feedback coming after this needs;
    // the rotation range is what maps a rim to a rack; the base's own tuning menu has no host-side
    // route on either platform without the vendor's SDK, and on this base the driver creates no
    // node for it at all. Each is a warning and a fallback, never a refusal to start — the string
    // views are literals and outlive everything that reads them.
    options.wanted = {CapabilityRequest{.capability = DeviceCapability::ConstantForce,
                                        .purpose = "force feedback",
                                        .fallback = "the wheel stays free"},
                      CapabilityRequest{.capability = DeviceCapability::ReadRotationRange,
                                        .purpose = "reading the wheel's own rotation range",
                                        .fallback = "the range this device's profile states is used"},
                      CapabilityRequest{.capability = DeviceCapability::TuningMenu,
                                        .purpose = "following the base's own tuning profile",
                                        .fallback = "the game's own settings stand alone"}};

    return options;
}

// What the write side is allowed to do, on the same test and for the same reason. The rate is the
// one the driver consumes rather than the one the transport advertises — this base's `hid-fanatec`
// runs its output on a two millisecond timer that is compile-time fixed, so the thousand hertz the
// interrupt endpoint's `bInterval` states is a ceiling nothing reaches. Pacing to the transport
// would be half the writes going nowhere.
[[nodiscard]] AudioOptions engineAudioOptions()
{
    auto options = AudioOptions{};

    // The same rule every device path here keeps, and the reason the golden frames do not move when
    // a machine gains a sound card.
    //
    // `RACEENGINE_AUDIO=1` overrides it, and only that way round: a run can ask for sound it would
    // not have had, and nothing can ask a gate for sound it must not have. That asymmetry is the
    // whole of why it is safe — the gates set neither variable and are unreachable from here.
    options.unattended =
        std::getenv("RACEENGINE_UNATTENDED") != nullptr || std::getenv("RACEENGINE_DUMP_FRAME") != nullptr;

    if (const auto* asked = std::getenv("RACEENGINE_AUDIO"); asked != nullptr && std::string_view(asked) == "1")
    {
        options.unattended = false;
    }

    // `RACEENGINE_DUMP_AUDIO` is `RACEENGINE_DUMP_FRAME` for sound, and it overrides unattended for
    // the same reason and by the same asymmetry: a run can ask to be recorded, and nothing can ask a
    // gate to make a noise. It opens no device — the mix goes to the file — so it is safe on a
    // machine somebody is working at, which is the whole point of it.
    if (const auto* path = std::getenv("RACEENGINE_DUMP_AUDIO"); path != nullptr && *path != '\0')
    {
        options.capturePath = path;
        options.unattended = false;
    }

    options.useFmod = true;

    return options;
}

[[nodiscard]] ForceFeedbackOptions engineForceFeedbackOptions()
{
    auto options = ForceFeedbackOptions{};

    options.unattended =
        std::getenv("RACEENGINE_UNATTENDED") != nullptr || std::getenv("RACEENGINE_DUMP_FRAME") != nullptr;
    options.outputHz = 500.0;
    // Five minutes of a 360 Hz publish, and the rate in that sentence is why the number moved
    // (2026-08-21). The stage-one trace is the deliverable rather than a diagnostic — it is recorded
    // whether or not a device is attached and whether or not anybody has asked for it — and what
    // makes it one is the *window*: a wheel complaint is settled by reading back the minutes before
    // the driver reached for the key. When the simulation moved onto its own thread the publish rate
    // tripled, so the 36000 frames that had been five minutes silently became a hundred seconds, and
    // both this comment and `~PlayerCar`'s said five. 96 bytes a frame, so the window costs 10.4 MB
    // against the 3.5 it did; that is the price of the instrument still being the one described.
    options.traceCapacity = 108000;

    // The pedals' motors, on for any set that has them. **Safe to state unconditionally**: the
    // profile only says a ClubSport V3 *would* have motors, and nothing is written until the device
    // link agrees they are attached and reachable — which on Linux means the pedals are on their own
    // USB cable, because a set wired through the base's RJ12 is only axes and the driver exposes no
    // control for it. On a rig with CSL pedals, or with V3 pedals wired the usual way, this costs a
    // boolean and produces silence.
    options.pedals.hasMotors = true;

    return options;
}

} // namespace

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
    sceneService(renderableEntityService, cameraService, sceneManagerService, shaderService),
    // Null: DirectInput's cooperative level is set against a window and this composition root has no
    // portable way to name one. A Windows build that wants exclusive access — which is what force
    // feedback needs there — passes its HWND here, and that is a one-line change to this call rather
    // than to the interface, which is the whole reason the parameter exists before its caller does.
    inputBackend(createInputBackend(nullptr)),
    inputService(*logger, *inputBackend, glfwWindow, engineInputOptions()),
    forceFeedbackService(*logger, inputService, engineForceFeedbackOptions()),
    audioService(*logger, engineAudioOptions())
{
    // An engine whose device would not come up has nothing left to do: every service below
    // was built against it, and there is no second backend to fall back to.
    // Thrown, not logged and thrown: main's boundary is what reports an engine that could not be
    // built, and saying it twice through two channels is how a reader ends up looking for two
    // problems.
    if (const auto initialised = renderer->init(); !initialised)
    {
        fail("Renderer initialisation failed: " + initialised.error());
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

                        // A pass that follows the window at a fraction of it keeps that fraction
                        // through the resize; see PostProcess::windowSizeDivisor.
                        const auto divisor = static_cast<int>(pass->windowSizeDivisor);
                        const auto passWidth = std::max(width / divisor, 1);
                        const auto passHeight = std::max(height / divisor, 1);
                        if (const auto recreated =
                                postProcessService.recreateOutputBuffer(postProcess, passWidth, passHeight);
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
        fail(physics.error());
    }
}

Engine::~Engine()
{
    // Only when something was written. A run with no wheel on it has nothing to report and a line
    // saying so every time is a line nobody reads.
    if (forceFeedbackService.writeRateHz() > 0.0)
    {
        logger->info("{}", forceFeedbackService.report());
    }

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

void Engine::onFrame(std::function<void()> callback)
{
    frameCallbacks.push_back(std::move(callback));
}

// One tick of simulation, always fixedTimeStep long. Order is writers before readers: the
// game's own logic, then the behaviour each entity carries, then the scene settling what
// both of them moved.
void Engine::update(float delta)
{
    RACEENGINE_ZONE_N("Engine::update");

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
    RACEENGINE_ZONE_N("Engine::step");

    frameDiagnostics.beginFrame();

    // The keyboard, once, on the thread that owns the window. GLFW's key state only changes inside
    // `glfwPollEvents` — which runs at the end of the previous step, inside `swapBuffers` — so this
    // reads exactly what a tick polling `keyPressed` for itself would have read, and reads it on the
    // one thread allowed to ask. Everything else about a device is taken by whoever is stepping the
    // simulation; see `InputService::sample`.
    inputService.pollWindow();

    const auto delta = frameDelta();

    // Clamping the accumulator, not the loop, is what bounds catch-up: time beyond the
    // budget is dropped here, so the loop can never find more than maxCatchUpSteps of work
    // and the leftover is always a fraction of one step.
    accumulator = std::min(accumulator + delta, fixedTimeStep * static_cast<float>(maxCatchUpSteps));

    // Counted, because it is the only clock anything temporal below may read. A capture run makes
    // the tick count a function of the frame number rather than of how fast the machine ran, and
    // an exposure adaptation driven by anything else would put a different image on disk.
    auto ticks = 0u;

    {
        RACEENGINE_ZONE_N("fixed steps");

        while (accumulator >= fixedTimeStep)
        {
            update(fixedTimeStep);
            accumulator -= fixedTimeStep;
            ticks++;
            simulatedTicks++;
        }
    }

    // Plotted rather than only counted: a frame that took none and a frame that took the catch-up
    // limit are the two ends of the spiral guard, and the shape of that channel is what says whether
    // the guard is being reached at all.
    RACEENGINE_PLOT("Fixed steps per frame", static_cast<double>(ticks));

    interpolationAlpha = accumulator / fixedTimeStep;

    {
        // The game's per-frame say, before anything this frame records is chosen: a callback here
        // may still write a camera's pose or hold a post-process pass, and both are read below.
        RACEENGINE_ZONE_N("frame callbacks");

        for (const auto& callback : frameCallbacks)
        {
            callback();
        }
    }

    {
        // No comma in the name, and that is not a style note: `tracy-csvexport` writes the zone name
        // into an unquoted CSV field, so a comma splits one row into columns that no longer line up.
        RACEENGINE_ZONE_N("cascade fit and camera matrices");

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
    }

    // The frame is the composition root's, start to finish: it opens once, every camera of
    // every scene records its view into it, the presenter records the pass that reaches the
    // screen, and one endFrame submits and presents the lot. Nothing below opens a frame of
    // its own, which is what makes N cameras N views inside one present rather than N
    // presents of which the first N-1 are empty.
    //
    // A backend that cannot open a frame — a swapchain gone out of date, a minimised window —
    // records nothing this step and skips the close, which is the one path with no present.
    //
    // Hoisted out of the `if` for one reason and it is the profiler's: `beginFrame` is where the CPU
    // waits on the frame already in flight, so it is the first place to look when a frame is long,
    // and a call inside a condition cannot carry a zone of its own.
    auto frameOpened = false;
    {
        RACEENGINE_ZONE_N("beginFrame (waits on the GPU)");
        frameOpened = renderer->beginFrame(static_cast<double>(simulatedTicks) * static_cast<double>(fixedTimeStep));
    }

    if (frameOpened)
    {
        {
            // The cascades first: a shadow cascade produces the depth map everything downstream
            // samples, and a producer recorded after its consumer is read before it is written (see
            // CameraRole). Separate passes rather than a sort because the order is fixed groups, and
            // a sort would allocate inside the frame to say so.
            RACEENGINE_ZONE_N("record shadow cascades");

            for (auto& scenePtr : sceneManagerService.getScenes())
            {
                if (!scenePtr)
                {
                    continue;
                }

                for (auto& camera : scenePtr->cameras)
                {
                    // A held cascade's map already shows the right picture — the shadow service
                    // kept its fit exactly where the map was rendered — so the frame spends
                    // nothing on it, which is most of what the far-cascade cache buys.
                    if (camera.role == CameraRole::ShadowCascade && !camera.contentsHeld)
                    {
                        renderer->recordView(*scenePtr, camera, delta);
                    }
                }
            }
        }

        // Then one probe's worth of capture, between the two: a probe shades from the cascades
        // recorded above, and the scene cameras below shade from the probe.
        recordProbeCaptures();

        {
            RACEENGINE_ZONE_N("record scene views");

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
                        // Immediately before the view that samples it, and inside the same frame:
                        // the occlusion is gathered from this camera's own geometry, so it is
                        // neither a group of its own above nor something a game could order for
                        // itself.
                        renderer->recordAmbientOcclusion(*scenePtr, camera, delta);
                        renderer->recordView(*scenePtr, camera, delta);

                        // Immediately after the view that filled it. The reduction this reads from
                        // is the tail of that view's post-process chain, so the copy has to be a
                        // command in the same frame and after those passes; what it copies reaches
                        // the CPU a fixed number of submissions later, never this one.
                        renderer->recordAutoExposure(camera);
                    }
                }
            }
        }

        presenterService.record();

        {
            RACEENGINE_ZONE_N("endFrame (submit and present)");
            renderer->endFrame();
        }
    }

    // The frame owns the report as it owns the frame: every recorder and the skinning path have
    // finished counting by here, and a reason is stated once rather than at each site that met it.
    frameDiagnostics.report();

    // After the present and before the swap: the capture reads what this frame put on screen.
    dumpFrameIfRequested();

    {
        RACEENGINE_ZONE_N("swapBuffers (polls events)");
        glfwWindow.swapBuffers();
    }

    // The frame's own boundary, last of all and after the swap, so Tracy's frame time is the whole
    // of `Engine::step` and matches what the window is actually showing.
    RACEENGINE_FRAME;
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
    RACEENGINE_ZONE_N("record probe capture");

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

    // Frame 120 unless told otherwise. The gates say nothing and get 120, which is what keeps a
    // golden frame a golden frame; `RACEENGINE_DUMP_FRAME_AT` is for looking at a moment the gate's
    // instant is too early to show — a car half a second into a launch has not turned yet, and
    // "which way did it go" is not a question frame 120 can answer.
    static const int dumpAt = []
    {
        const auto* at = std::getenv("RACEENGINE_DUMP_FRAME_AT");
        const auto asked = at == nullptr ? 0 : std::atoi(at);

        return asked > 0 ? asked : 120;
    }();

    static int dumpFrameCount = 0;
    if (++dumpFrameCount < dumpAt)
    {
        return;
    }

    stopRequested = true;

    // Before the frame capture rather than after it: `captureFrame` replays the presenter's pass
    // onto a freshly acquired swapchain image, which runs the post chain again, and the buffers
    // wanted here are the ones this frame's passes left behind.
    if (const char* bufferPrefix = std::getenv("RACEENGINE_DUMP_BUFFERS"); bufferPrefix != nullptr)
    {
        if (const auto dumped = renderer->captureBuffers(bufferPrefix); !dumped)
        {
            logger->error("Attachment dump beside {} did not produce files: {}", bufferPrefix, dumped.error());
            exitStatus = 1;
        }
    }

    if (const auto captured = renderer->captureFrame(dumpPath); !captured)
    {
        logger->error("Frame capture to {} did not produce a file: {}", dumpPath, captured.error());
        exitStatus = 1;
    }
}

} // namespace raceengine
