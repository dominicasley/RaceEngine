// FrameDiagnostics bodies. Declarations are in Graphics/Api/FrameDiagnostics.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include <spdlog/logger.h>

module raceengine.graphics;

import :FrameDiagnostics;

namespace raceengine
{

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
