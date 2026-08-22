// Precompiled header for RaceEngine and RaceEngineSandbox. Nothing includes this file — CMake
// hands it to the compiler ahead of every translation unit in those targets.
//
// It exists for one measured reason: a standard header textually included in a *global module
// fragment* has to be merged against the copy already inside every imported BMI, and in a unit
// that imports `raceengine` that merge dominates the compile. Precompiling the headers once
// removes it. One probe unit, best of three, 2026-08-22: 11.13 s without this, 1.74 s with it,
// and the unit's own #includes unchanged. A unit that includes none of these pays 0.18 s.
//
// **What belongs here**: a standard header that this project's global module fragments already
// name. Nothing else. In particular NOT first-party headers (there are none — this is a modules
// project), and not glm, which measures at the baseline and would only make the PCH bigger.
//
// **What does not belong here**: anything whose macros could change meaning per target. A PCH is
// compiled once per target with that target's flags, so a header reading FMOD_* or RACEENGINE_*
// state must stay where it is.
//
// The two groups below are split by what they cost *alone*; both are worth carrying, because the
// cost saturates — six headers from the second group together still cost ~5.7 s, which the PCH
// also takes away. Full account and the per-header table: docs/build-times.md.

// Expensive on their own, in a unit that imports the engine (+2 s to +8 s each).
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <thread>

// Free on their own, but not in combination, and every one of them is named by a global module
// fragment somewhere in this tree.
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <expected>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
