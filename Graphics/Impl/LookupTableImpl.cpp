// LookupTable bodies. Declarations are in Graphics/Api/LookupTable.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

module raceengine.graphics;

import :LookupTable;

namespace raceengine
{

} // namespace raceengine
