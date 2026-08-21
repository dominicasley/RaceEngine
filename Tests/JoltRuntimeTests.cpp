#include <catch2/catch_test_macros.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::tearDownJolt;

// This case earns its place by linking rather than by asserting. The physics backend is a static
// library and Jolt is another one behind it, so until something actually calls across the bridge
// the linker has no undefined symbol to resolve, never opens either archive, and "Jolt is wired up"
// stays a claim about CMake rather than a fact about the binary. Calling it here is what makes the
// build prove it.
TEST_CASE("the physics backend starts and stops", "[physics][jolt]")
{
    const auto started = bringUpJolt();
    REQUIRE(started.has_value());

    // Jolt's factory and type registry are process-wide, so standing them up twice is a caller
    // error rather than a no-op, and it is reported as one.
    const auto again = bringUpJolt();
    REQUIRE_FALSE(again.has_value());

    tearDownJolt();

    // Balanced, so the case leaves the process as it found it and can run in any order beside the
    // rest of the suite. Tearing down twice is harmless by design: an owner that failed halfway
    // through construction unwinds through it.
    tearDownJolt();

    REQUIRE(bringUpJolt().has_value());
    tearDownJolt();
}
