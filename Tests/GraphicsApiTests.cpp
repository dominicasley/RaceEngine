#include <cstdlib>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>

import raceengine;
import raceengine.tests.log;

using raceengine::GraphicsApi;
using raceengine::selectGraphicsApi;
using raceengine::tests::CapturedLog;

namespace
{

// RACEENGINE_RENDERER is process state, and Catch2 runs every case in one process, so each case
// has to put it back. Restoring the previous value rather than clearing it means a run launched
// with the variable already set still behaves the same afterwards.
class RendererEnvironment
{
public:
    explicit RendererEnvironment(const std::optional<std::string>& requested)
    {
        const char* existing = std::getenv("RACEENGINE_RENDERER");
        if (existing != nullptr)
        {
            previous = std::string(existing);
        }

        set(requested);
    }

    RendererEnvironment(const RendererEnvironment&) = delete;
    RendererEnvironment(RendererEnvironment&&) = delete;
    RendererEnvironment& operator=(const RendererEnvironment&) = delete;
    RendererEnvironment& operator=(RendererEnvironment&&) = delete;

    ~RendererEnvironment()
    {
        set(previous);
    }

private:
    static void set(const std::optional<std::string>& value)
    {
#ifdef _WIN32
        static_cast<void>(_putenv_s("RACEENGINE_RENDERER", value.has_value() ? value.value().c_str() : ""));
#else
        if (value.has_value())
        {
            static_cast<void>(setenv("RACEENGINE_RENDERER", value.value().c_str(), 1));
        }
        else
        {
            static_cast<void>(unsetenv("RACEENGINE_RENDERER"));
        }
#endif
    }

    std::optional<std::string> previous{};
};

} // namespace

TEST_CASE("an unset RACEENGINE_RENDERER selects Vulkan without complaint", "[api]")
{
    CapturedLog log;
    const RendererEnvironment environment(std::nullopt);

    REQUIRE(selectGraphicsApi(log.sink()) == GraphicsApi::Vulkan);
    REQUIRE(log.occurrences("Selected graphics API: Vulkan") == 1);
    REQUIRE(log.occurrences("Unknown RACEENGINE_RENDERER") == 0);
}

TEST_CASE("RACEENGINE_RENDERER=opengl selects OpenGL", "[api]")
{
    CapturedLog log;
    const RendererEnvironment environment(std::string("opengl"));

    REQUIRE(selectGraphicsApi(log.sink()) == GraphicsApi::OpenGL);
    REQUIRE(log.occurrences("Selected graphics API: OpenGL") == 1);
    REQUIRE(log.occurrences("Selected graphics API: Vulkan") == 0);
}

TEST_CASE("RACEENGINE_RENDERER=vulkan selects Vulkan without complaint", "[api]")
{
    CapturedLog log;
    const RendererEnvironment environment(std::string("vulkan"));

    REQUIRE(selectGraphicsApi(log.sink()) == GraphicsApi::Vulkan);
    REQUIRE(log.occurrences("Unknown RACEENGINE_RENDERER") == 0);
}

TEST_CASE("an unrecognised RACEENGINE_RENDERER warns and falls back to Vulkan", "[api]")
{
    CapturedLog log;
    const RendererEnvironment environment(std::string("metal"));

    REQUIRE(selectGraphicsApi(log.sink()) == GraphicsApi::Vulkan);
    REQUIRE(log.occurrences("Unknown RACEENGINE_RENDERER value 'metal'; using Vulkan") == 1);
    REQUIRE(log.occurrences("Selected graphics API: Vulkan") == 1);
}

TEST_CASE("the match is exact, so a differently cased value is not the backend it names", "[api]")
{
    CapturedLog log;
    const RendererEnvironment environment(std::string("OpenGL"));

    // Worth pinning rather than assuming: the gates set the variable themselves, so nothing else
    // would notice if "OpenGL" started meaning OpenGL — or stopped warning that it does not.
    REQUIRE(selectGraphicsApi(log.sink()) == GraphicsApi::Vulkan);
    REQUIRE(log.occurrences("Unknown RACEENGINE_RENDERER value 'OpenGL'; using Vulkan") == 1);
}

TEST_CASE("an empty RACEENGINE_RENDERER warns rather than being read as unset", "[api]")
{
    CapturedLog log;
    const RendererEnvironment environment(std::string(""));

    REQUIRE(selectGraphicsApi(log.sink()) == GraphicsApi::Vulkan);
    REQUIRE(log.occurrences("Unknown RACEENGINE_RENDERER") == 1);
}
