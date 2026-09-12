// Engine bodies. Declarations are in Engine.cppm.
//
// A **module implementation unit** — `module raceengine;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. This is what this file's
// `module :private;` fragment used to be, and it is here because that fragment did not do the job:
// measured, editing inside one still changed the BMI and still rebuilt every importer.
// Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

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
    // **Ten minutes of a 360 Hz publish**, on Dominic's instruction (2026-08-28).
    //
    // The stage-one trace is the deliverable rather than a diagnostic — it is recorded whether or not
    // a device is attached and whether or not anybody has asked for it — and what makes it one is the
    // *window*: a wheel complaint is settled by reading back the minutes before the driver reached for
    // the key.
    //
    // **The window has now been overrun twice by real drives and each time it cost the cold start.**
    // A thermal or pressure model is a statement about the first few minutes of a session, so a ring
    // that keeps the *last* five minutes of a nine-minute stint throws away precisely the part those
    // models are about: the 2026-08-28 stage-2 lap ran 545 s and its trace began at t = 245.
    //
    // It moved once before for the opposite reason (2026-08-21): the simulation went onto its own
    // thread, the publish rate tripled, and 36000 frames that had been five minutes silently became a
    // hundred seconds while both this comment and `~PlayerCar`'s went on saying five.
    //
    // **The cost is memory and it is no longer small.** This comment used to say 96 bytes a frame and
    // 10.4 MB, which was true when a frame was a dozen doubles; a frame now carries eighteen channels
    // per corner and comes to about **768 bytes**, so ten minutes is roughly **166 MB** resident
    // against 83 for five. That is the price of a session-length instrument and it is stated here so
    // the next person to widen it knows what it costs.
    options.traceCapacity = 216000;

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
    occlusionCullingService(*logger, memoryStorageService, fboService, postProcessService),
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
namespace
{

// ---- The probe scheduler ------------------------------------------------------------------------
//
// Which probe the backend advances this frame, and which probes hold the specular pool. Both are
// decided against one view — the scene's first Scene camera — because a city carries hundreds of
// probes against a pool of `maxIblProbes` slices, and the slices that matter are the ones nearest
// the eye.
//
// The pool used to be handed out once, to the first eight probes captured, and held for the
// probe's life. On a city whose probes are ordered outwards from the grid slot that was the global
// probe and the seven street probes nearest spawn, so every reflection more than a few stands from
// the start was the sky over the start line: the buildings were in every probe's photograph and
// in no car's paint (2026-09-13, the seat: "cars are just using the default probe"). Now a pool
// slice follows the view — when a probe the view wants holds none and a probe it does not want
// holds one, the slice changes hands and the newcomer re-captures into it.
//
// Three rules keep the backend's guarantees, and each is load-bearing:
// - **A capture in flight is always the probe advanced, and the pool does not change while one
//   is.** The backend has one readback buffer and one scratch slice, and "the probe in flight is
//   the next one advanced" is what makes both safe (recordProbeCapture).
// - **At most one slice changes hands a frame, and the probe that received it is the probe
//   advanced.** Its first face puts it in Capturing before any view of this frame is recorded, and
//   uploadProbes reads no reflection off a Capturing probe, so the frame never reflects a slice
//   whose photograph belongs to another street.
// - **A scene that fits the pool is scheduled exactly as before**: the first not-Ready probe in
//   scene order, every probe keeping the slice its first capture gave it. That is what keeps both
//   frame gates still — the apron places three probes and the circuit one.

// A probe's standing with a view: distance squared, and -1 for the global probe so it is wanted
// before everything and evicted never. The same rank uploadProbes uses to choose a frame's eight.
[[nodiscard]] float probeRank(const LightProbe& probe, const glm::vec3& viewPosition)
{
    if (probe.global)
    {
        return -1.0f;
    }

    const auto offset = probe.position - viewPosition;
    return glm::dot(offset, offset);
}

// The probes a view shades from: the global one and the nearest locals up to the pool's size,
// nearest first. Unlike uploadProbes this does not skip a probe that has never been captured — a
// probe the view wants and has no photograph of is exactly the one to capture next.
struct WantedProbes
{
    std::array<LightProbe*, maxIblProbes> probes{};
    std::array<float, maxIblProbes> ranks{};
    std::size_t count = 0;

    [[nodiscard]] bool contains(const LightProbe& probe) const
    {
        for (auto index = std::size_t{0}; index < count; index++)
        {
            if (probes[index] == &probe)
            {
                return true;
            }
        }

        return false;
    }
};

[[nodiscard]] WantedProbes wantedProbes(Scene& scene, const glm::vec3& viewPosition)
{
    auto wanted = WantedProbes{};

    for (auto& probe : scene.probes)
    {
        const auto rank = probeRank(probe, viewPosition);

        if (wanted.count < wanted.probes.size())
        {
            wanted.probes[wanted.count] = &probe;
            wanted.ranks[wanted.count] = rank;
            wanted.count++;
            continue;
        }

        auto worst = std::size_t{0};
        for (auto index = std::size_t{1}; index < wanted.count; index++)
        {
            if (wanted.ranks[index] > wanted.ranks[worst])
            {
                worst = index;
            }
        }

        if (rank < wanted.ranks[worst])
        {
            wanted.probes[worst] = &probe;
            wanted.ranks[worst] = rank;
        }
    }

    // Nearest first, so the street the car is standing in is the first one photographed — on a
    // cold bake, and again after a time-of-day change dirties the whole graph.
    for (auto outer = std::size_t{1}; outer < wanted.count; outer++)
    {
        for (auto inner = outer; inner > 0 && wanted.ranks[inner] < wanted.ranks[inner - 1]; inner--)
        {
            std::swap(wanted.probes[inner], wanted.probes[inner - 1]);
            std::swap(wanted.ranks[inner], wanted.ranks[inner - 1]);
        }
    }

    return wanted;
}

// How much nearer than the holder a newcomer must stand before a slice changes hands, as a ratio
// of distance squared: 0.64 is a fifth of the holder's distance. Without it a car parked between
// two stands would trade the same slice back and forth, eight frames of capture each way, for as
// long as it sat there. The stands are sixty metres apart and the pool's furthest member on a
// straight street is three stands out, so a fifth is a dozen metres of travel — under a second.
constexpr auto sliceHandoverRankRatio = 0.64f;

// Moves one pool slice from the probe the view wants least to a probe it wants and that holds
// none, and returns the newcomer; nullptr when nothing moved. Only a probe outside the wanted set
// gives its slice up, so a scene that fits the pool never moves one, and a wanted probe with no
// slice while the pool still has spares is left alone: it is Dirty or will be captured in its
// turn, and the backend hands a fresh slice to a capture while it has any.
[[nodiscard]] LightProbe* handOverSlice(Scene& scene, const WantedProbes& wanted, const glm::vec3& viewPosition)
{
    for (auto index = std::size_t{0}; index < wanted.count; index++)
    {
        auto& newcomer = *wanted.probes[index];
        if (newcomer.arraySlice.has_value())
        {
            continue;
        }

        LightProbe* holder = nullptr;
        auto holderRank = 0.0f;
        for (auto& probe : scene.probes)
        {
            // Nothing is in flight here, so a held slice is a pool slice and never the scratch.
            if (!probe.arraySlice.has_value() || wanted.contains(probe))
            {
                continue;
            }

            const auto rank = probeRank(probe, viewPosition);
            if (holder == nullptr || rank > holderRank)
            {
                holder = &probe;
                holderRank = rank;
            }
        }

        if (holder == nullptr)
        {
            return nullptr;
        }

        if (wanted.ranks[index] > holderRank * sliceHandoverRankRatio)
        {
            continue;
        }

        // The holder keeps its irradiance and its state: it shades diffuse as it did, and takes its
        // reflection from the global probe as a probe past the pool always has. The newcomer keeps
        // its irradiance too — the cache's or its last capture's — and re-photographs into the slice.
        newcomer.arraySlice = holder->arraySlice;
        holder->arraySlice.reset();
        newcomer.invalidate();
        return &newcomer;
    }

    return nullptr;
}

// The probe to advance this frame, or nullptr when the scene's graph is settled.
[[nodiscard]] LightProbe* nextProbeCapture(Scene& scene, const Camera* view)
{
    for (auto& probe : scene.probes)
    {
        if (probe.state == LightProbeState::Capturing || probe.state == LightProbeState::Projecting)
        {
            return &probe;
        }
    }

    // A scene the pool holds whole, or a scene with no view to follow: the first not-Ready probe in
    // scene order, which is the rule there has always been.
    if (view == nullptr || scene.probes.size() <= maxIblProbes)
    {
        for (auto& probe : scene.probes)
        {
            if (probe.state != LightProbeState::Ready)
            {
                return &probe;
            }
        }

        return nullptr;
    }

    const auto wanted = wantedProbes(scene, view->position);

    if (auto* newcomer = handOverSlice(scene, wanted, view->position); newcomer != nullptr)
    {
        return newcomer;
    }

    for (auto index = std::size_t{0}; index < wanted.count; index++)
    {
        if (wanted.probes[index]->state != LightProbeState::Ready)
        {
            return wanted.probes[index];
        }
    }

    for (auto& probe : scene.probes)
    {
        if (probe.state != LightProbeState::Ready)
        {
            return &probe;
        }
    }

    return nullptr;
}

} // namespace

void Engine::recordProbeCaptures()
{
    RACEENGINE_ZONE_N("record probe capture");

    for (auto& scenePtr : sceneManagerService.getScenes())
    {
        if (!scenePtr)
        {
            continue;
        }

        // The view the pool follows: the scene's first Scene camera. In the layered frame that is
        // the world camera, and the car and frame cameras share its pose; the mirror camera stands
        // at the driver's eye, two metres from it, which no stand can tell apart.
        const Camera* view = nullptr;
        for (const auto& camera : scenePtr->cameras)
        {
            if (camera.role == CameraRole::Scene)
            {
                view = &camera;
                break;
            }
        }

        auto* probe = nextProbeCapture(*scenePtr, view);
        if (probe == nullptr)
        {
            continue;
        }

        renderer->recordProbeCapture(*scenePtr, *probe);

        return;
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
