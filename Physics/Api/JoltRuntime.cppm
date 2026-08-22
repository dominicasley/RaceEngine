module;

#include <expected>
#include <string>

export module raceengine.physics:JoltRuntime;

// The physics backend's process-wide bring-up, declared rather than included: the definitions live
// in Physics/Backend/JoltBackend.cpp, which is a separate target precisely so that Jolt's
// instruction-set usage requirements never reach a module unit. See that file for what goes wrong
// otherwise. Declared `extern "C++"` alongside a matching definition is the same bridge
// GLTFService.cppm uses for the tinygltf loader.
extern "C++" bool raceengineJoltBringUp(std::string& reason);
extern "C++" void raceengineJoltTearDown();

namespace raceengine
{

// Jolt's allocator, factory and type registry are global and have to be stood up before any Jolt
// object exists and torn down after the last one dies. That makes this the composition root's
// business and nobody else's, which is why it is a pair of free functions rather than something a
// service owns: two owners of a process-wide singleton is the bug it would be hiding.
export [[nodiscard]] std::expected<void, std::string> bringUpJolt();

// Safe to call without a matching bring-up, so a partially constructed owner can unwind through it.
export void tearDownJolt();

} // namespace raceengine
