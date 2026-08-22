// DeviceProfile bodies. Declarations are in Input/Api/DeviceProfile.cppm.
//
// A **module implementation unit** — `module raceengine.input;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

module raceengine.input;

import :DeviceProfile;
import :InputBackend;

namespace raceengine
{

} // namespace raceengine
