module;

#include <cstdlib>
#include <string_view>

#include <spdlog/logger.h>

export module raceengine.graphics:GraphicsApi;

namespace raceengine
{

export enum class GraphicsApi { OpenGL, Vulkan };

// Resolved before the window exists: the window needs the api to pick its client API
// (GL context vs GLFW_NO_API), and the renderer factory needs it afterwards.
export [[nodiscard]] GraphicsApi selectGraphicsApi(spdlog::logger& logger)
{
    const char* requested = std::getenv("RACEENGINE_RENDERER");

    if (requested != nullptr && std::string_view(requested) == "opengl")
    {
        logger.info("Selected graphics API: OpenGL");
        return GraphicsApi::OpenGL;
    }

    if (requested != nullptr && std::string_view(requested) != "vulkan")
    {
        logger.warn("Unknown RACEENGINE_RENDERER value '{}'; using Vulkan", requested);
    }

    logger.info("Selected graphics API: Vulkan");
    return GraphicsApi::Vulkan;
}

} // namespace raceengine
