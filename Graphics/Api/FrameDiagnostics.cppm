module;

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include <spdlog/logger.h>

export module raceengine.graphics:FrameDiagnostics;

namespace raceengine
{

// Why a frame skipped work it was asked to do.
//
// One list for both backends, because the reasons are properties of the engine's own model — a
// renderable naming an unloaded asset, a primitive with no material, a ring with no slot left —
// rather than of an API. Each backend used to carry a private `bool …Logged` (or, worse, a
// function-local `static`, which is per *process* and so shared between two renderers) per
// reason, purely to keep a per-primitive condition from filling the log. That is an
// infrastructure investment in log-and-continue: the throttle was bespoke at every site and the
// reason itself was thrown away.
//
// A skipped draw is a fact about the frame, so it is counted here and the frame reports it once:
// the first frame in which a reason appears logs it with a count and the first instance's
// detail, later frames add to the tally silently, and the tally is stated again at shutdown.
// Nothing on the draw path allocates or formats — record() is a compare and two increments, and
// the detail lambda runs only for the first occurrence a reason ever has.
export enum class FrameDiagnostic : size_t {
    StaleModelHandle,
    StaleMeshHandle,
    MeshNotUploaded,
    MeshBufferNotUploaded,
    PrimitiveWithoutMaterial,
    PrimitiveNotUploaded,
    MissingCameraOutput,
    JointLimitExceeded,
    LightLimitExceeded,
    SkinningRejected,
    ScenePipelineUnavailable,
    DescriptorSetUnavailable,
    DrawDataRingExhausted,
    JointDataRingExhausted,
    FrameDataRingExhausted,
    PostProcessSkipped,
    PostProcessInputsExceeded,
    PresentPassSkipped,
    UnsupportedTopology,
    UnsupportedIndexType,
    UnsupportedTextureLayout,
    MipGenerationUnavailable,
    ShadowCascadeUnavailable,
    ViewShaderUnavailable,
    ProbeLimitExceeded,
    ProbeCaptureSkipped,
    ExposureMeterSkipped,
    AmbientOcclusionSkipped,
    ColourGradeUnavailable,
};

export inline constexpr size_t frameDiagnosticCount = static_cast<size_t>(FrameDiagnostic::ColourGradeUnavailable) + 1;

// Reads as the tail of "<n> …", so every phrase is a countable noun.
export [[nodiscard]] constexpr const char* describe(const FrameDiagnostic diagnostic)
{
    switch (diagnostic)
    {
    case FrameDiagnostic::StaleModelHandle:
        return "renderable(s) naming a model or shader that is no longer loaded";
    case FrameDiagnostic::StaleMeshHandle:
        return "renderable mesh(es) naming a mesh that is no longer loaded";
    case FrameDiagnostic::MeshNotUploaded:
        return "mesh(es) still without a GPU resource after an upload";
    case FrameDiagnostic::MeshBufferNotUploaded:
        return "primitive(s) whose vertex or index buffer was never uploaded";
    case FrameDiagnostic::PrimitiveWithoutMaterial:
        return "primitive(s) with no material and shader";
    case FrameDiagnostic::PrimitiveNotUploaded:
        return "primitive(s) with no GPU binding";
    case FrameDiagnostic::MissingCameraOutput:
        return "camera(s) whose output framebuffer has no colour or depth target";
    case FrameDiagnostic::JointLimitExceeded:
        return "draw(s) supplying more joints than the shader array holds";
    case FrameDiagnostic::LightLimitExceeded:
        return "view(s) declaring more lights than the shader array holds";
    case FrameDiagnostic::SkinningRejected:
        return "mesh(es) whose animation sampling ozz would not run";
    case FrameDiagnostic::ScenePipelineUnavailable:
        return "draw(s) with no scene pipeline to record against";
    case FrameDiagnostic::DescriptorSetUnavailable:
        return "draw(s) or pass(es) the descriptor pool had no room for";
    case FrameDiagnostic::DrawDataRingExhausted:
        return "draw(s) past the end of the draw-data ring";
    case FrameDiagnostic::JointDataRingExhausted:
        return "skinned draw(s) drawn in their bind pose, the joint-palette ring having no slot left";
    case FrameDiagnostic::FrameDataRingExhausted:
        return "view(s) past the end of the frame-data ring";
    case FrameDiagnostic::PostProcessSkipped:
        return "post-process pass(es) wanting an input, a colour output or a fullscreen shader";
    case FrameDiagnostic::PostProcessInputsExceeded:
        return "post-process pass(es) declaring more inputs than the fullscreen set carries";
    case FrameDiagnostic::PresentPassSkipped:
        return "present pass(es) skipped, leaving the clear colour on screen";
    case FrameDiagnostic::UnsupportedTopology:
        return "primitive(s) using a draw mode this backend has no topology for";
    case FrameDiagnostic::UnsupportedIndexType:
        return "primitive(s) using an index component type this backend does not carry";
    case FrameDiagnostic::UnsupportedTextureLayout:
        return "texture(s) carrying less pixel data than their declared layout needs";
    case FrameDiagnostic::MipGenerationUnavailable:
        return "sampled image(s) uploaded without mips, the format having no linear blit";
    case FrameDiagnostic::ShadowCascadeUnavailable:
        return "view(s) shading lit, the cascade depth map they asked for having no target";
    case FrameDiagnostic::ViewShaderUnavailable:
        return "view(s) recording nothing, the shader they draw every entity with being unloaded";
    case FrameDiagnostic::ProbeLimitExceeded:
        return "light probe(s) past the end of the probe array, lighting nothing";
    case FrameDiagnostic::ProbeCaptureSkipped:
        return "light probe capture(s) abandoned, the target or the prefilter pipeline being unavailable";
    case FrameDiagnostic::ExposureMeterSkipped:
        return "exposure meter reading(s) not taken, the reduction chain having no image to copy from";
    case FrameDiagnostic::AmbientOcclusionSkipped:
        return "view(s) shaded without their occlusion, its buffer having no image to sample";
    case FrameDiagnostic::ColourGradeUnavailable:
        return "frame(s) presented ungraded, the presenter's lookup table having no image to sample";
    }

    return "skipped item(s) of an unnamed kind";
}

// The engine's one tally, owned by the composition root and handed to everything that can skip
// work: both backends and the skinning path. Per Engine, not per process — two engines in one
// test runner keep their own counts, which the `static` latches this replaced could not.
export class FrameDiagnostics
{
private:
    struct Entry
    {
        unsigned int inFrame = 0;
        unsigned int total = 0;
        std::string detail;
        bool reported = false;
    };

    spdlog::logger& logger;
    std::array<Entry, frameDiagnosticCount> entries{};
    unsigned int frames = 0;

    [[nodiscard]] Entry& entry(const FrameDiagnostic diagnostic)
    {
        return entries[static_cast<size_t>(diagnostic)];
    }

public:
    explicit FrameDiagnostics(spdlog::logger& logger);
    FrameDiagnostics(const FrameDiagnostics&) = delete;
    FrameDiagnostics(FrameDiagnostics&&) = delete;
    FrameDiagnostics& operator=(const FrameDiagnostics&) = delete;
    FrameDiagnostics& operator=(FrameDiagnostics&&) = delete;
    ~FrameDiagnostics();

    // Starts a frame's tally. Totals survive it; anything recorded before the first frame — an
    // upload, a framebuffer the backend could not serve — is carried into the first report.
    void beginFrame();

    void record(FrameDiagnostic diagnostic);

    // The detail is what the site would have put in its one-off log line. It is built only for
    // the first occurrence a reason ever has, so the per-primitive path pays a branch rather
    // than a string.
    template <typename Detail> void record(const FrameDiagnostic diagnostic, Detail&& describeFirst)
    {
        auto& counted = entry(diagnostic);

        if (counted.total == 0)
        {
            counted.detail = std::forward<Detail>(describeFirst)();
        }

        counted.inFrame++;
        counted.total++;
    }

    // One line per frame at most, naming only reasons that have not been reported before.
    void report();

    [[nodiscard]] unsigned int count(FrameDiagnostic diagnostic) const;

private:
    [[nodiscard]] static std::string join(const std::string& summary, const std::string& item)
    {
        return summary.empty() ? item : summary + "; " + item;
    }
};

FrameDiagnostics::FrameDiagnostics(spdlog::logger& logger) :
    logger(logger)
{
}

FrameDiagnostics::~FrameDiagnostics()
{
    std::string summary;

    for (size_t index = 0; index < entries.size(); index++)
    {
        if (entries[index].total == 0)
        {
            continue;
        }

        summary =
            join(summary, std::to_string(entries[index].total) + " " + describe(static_cast<FrameDiagnostic>(index)));
    }

    if (summary.empty())
    {
        return;
    }

    logger.warn("Skipped work over {} frame(s): {}", frames, summary);
}

void FrameDiagnostics::beginFrame()
{
    frames++;

    for (auto& counted : entries)
    {
        counted.inFrame = 0;
    }
}

void FrameDiagnostics::record(const FrameDiagnostic diagnostic)
{
    auto& counted = entry(diagnostic);
    counted.inFrame++;
    counted.total++;
}

void FrameDiagnostics::report()
{
    std::string summary;

    for (size_t index = 0; index < entries.size(); index++)
    {
        auto& counted = entries[index];
        if (counted.total == 0 || counted.reported)
        {
            continue;
        }

        // inFrame is zero only for something recorded before any frame opened — an upload or a
        // framebuffer creation — in which case the total is the whole story.
        const auto scope = counted.inFrame > 0 ? counted.inFrame : counted.total;
        auto item = std::to_string(scope) + " " + describe(static_cast<FrameDiagnostic>(index));

        if (!counted.detail.empty())
        {
            item += " (first: " + counted.detail + ")";
        }

        summary = join(summary, item);
        counted.reported = true;
    }

    if (summary.empty())
    {
        return;
    }

    logger.warn("Skipped work in frame {}: {}. Recurrences are counted, not repeated.", frames, summary);
}

unsigned int FrameDiagnostics::count(const FrameDiagnostic diagnostic) const
{
    return entries[static_cast<size_t>(diagnostic)].total;
}

} // namespace raceengine
