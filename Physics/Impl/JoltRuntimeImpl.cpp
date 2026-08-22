// JoltRuntime bodies. Declarations are in Api/JoltRuntime.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <string>

module raceengine.physics;

// The bridge declarations this file calls through. Repeated here rather than imported: they sit in
// the partition's purview and are not exported, so an implementation unit of the same module cannot
// see them through the primary interface. Declaring them here puts them in the same module, which is
// what makes the symbol the one Physics/Backend/JoltBackend.cpp defines.
extern "C++" bool raceengineJoltBringUp(std::string& reason);
extern "C++" void raceengineJoltTearDown();

namespace raceengine
{

[[nodiscard]] std::expected<void, std::string> bringUpJolt()
{
    auto reason = std::string();
    if (!raceengineJoltBringUp(reason))
    {
        return std::unexpected("the physics backend would not start: " + reason);
    }

    return {};
}

void tearDownJolt()
{
    raceengineJoltTearDown();
}

} // namespace raceengine
