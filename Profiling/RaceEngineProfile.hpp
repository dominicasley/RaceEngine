// Tracy zone markers for RaceEngine and RaceEngineSandbox.
//
// **This is a header in a modules project, and that is deliberate rather than an oversight.** A
// profiler zone has to know the file, line and function it was written at, which Tracy captures
// with `__FILE__`, `__LINE__` and `__FUNCTION__` inside a macro — and **macros do not cross an
// `import`**. There is no module that can export `ZoneScoped`. The alternative is Tracy's C API,
// which allocates a source location per zone entry at run time and would put an allocation in the
// 360 Hz tick this file exists to measure. So: one header, included from a *global module
// fragment* like any third-party header, and named unambiguously because `RaceEngine/` is on the
// public include path.
//
// **It costs nothing when the profiler is off.** `RACEENGINE_HAS_TRACY` is defined only by
// `RACEENGINE_WITH_TRACY=ON`, and without it this file includes nothing at all — so a unit that
// carries zone markers compiles identically in `build/dev`, and neither parity gate can see them.
//
// Build and run it with the `profile` preset; see docs/profiling.md.

#pragma once

#if defined(RACEENGINE_HAS_TRACY)

#include <tracy/Tracy.hpp>

// A scope's own duration, named after the enclosing function or after the string given. The string
// must be a literal: Tracy stores the pointer rather than the characters.
#define RACEENGINE_ZONE ZoneScoped
#define RACEENGINE_ZONE_N(name) ZoneScopedN(name)
#define RACEENGINE_ZONE_NC(name, colour) ZoneScopedNC(name, colour)

// The end of a frame on Tracy's main timeline, and the end of a *named* frame on a track of its
// own — which is how the simulation thread's 360 Hz tick gets a cadence plot separate from the
// render frame's.
#define RACEENGINE_FRAME FrameMark
#define RACEENGINE_FRAME_N(name) FrameMarkNamed(name)

// A numeric channel over time, plotted under the timeline.
#define RACEENGINE_PLOT(name, value) TracyPlot(name, value)

// Names the calling thread in the profiler. Call it once, from the thread itself.
#define RACEENGINE_THREAD(name) ::tracy::SetThreadName(name)

#else

// Every marker evaluates its arguments and produces nothing. Evaluating them is what keeps a
// profiled build and an unprofiled one agreeing about side effects, and what stops `-Wunused-*`
// from firing on a name only the profiler reads.
#define RACEENGINE_ZONE static_cast<void>(0)
#define RACEENGINE_ZONE_N(name) static_cast<void>(name)
#define RACEENGINE_ZONE_NC(name, colour)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        static_cast<void>(name);                                                                                       \
        static_cast<void>(colour);                                                                                     \
    } while (false)
#define RACEENGINE_FRAME static_cast<void>(0)
#define RACEENGINE_FRAME_N(name) static_cast<void>(name)
#define RACEENGINE_PLOT(name, value)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        static_cast<void>(name);                                                                                       \
        static_cast<void>(value);                                                                                      \
    } while (false)
#define RACEENGINE_THREAD(name) static_cast<void>(name)

#endif
