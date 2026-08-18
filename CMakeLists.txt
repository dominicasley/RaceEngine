set(CMAKE_CXX_STANDARD 23)
cmake_minimum_required (VERSION 3.28)

# Warning/sanitizer flags ride an INTERFACE carrier so vendored code stays untouched.
add_library(raceengine_warnings INTERFACE)
target_compile_options(raceengine_warnings INTERFACE
        -Werror
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion -Wimplicit-fallthrough)
option(RACEENGINE_SANITIZE "Build first-party code with ASan+UBSan" OFF)
if(RACEENGINE_SANITIZE)
    target_compile_options(raceengine_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(raceengine_warnings INTERFACE -fsanitize=address,undefined)
endif()

find_package(spdlog CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED)
find_path(STB_INCLUDE_DIRS "stb_image.h" REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h" REQUIRED)

include(FetchContent)
set(ozz_build_tools OFF CACHE BOOL "" FORCE)
set(ozz_build_samples OFF CACHE BOOL "" FORCE)
set(ozz_build_tests OFF CACHE BOOL "" FORCE)
set(ozz_build_howtos OFF CACHE BOOL "" FORCE)
set(ozz_build_gltf OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        ozz_animation
        GIT_REPOSITORY https://github.com/guillaumeblanc/ozz-animation
        GIT_TAG 0.17.0
        GIT_SHALLOW TRUE
        PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PatchOzzCMakeMinimum.cmake)
FetchContent_MakeAvailable(ozz_animation)

add_library(glad STATIC vendor/glad/src/gl.c)
target_include_directories(glad SYSTEM PUBLIC vendor/glad/include)

add_library (Engine STATIC
        Io/ThirdPartyImpl.cpp)

target_sources(Engine PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        Engine.cppm
        Async/Async.cppm
        Game/Game.cppm
        Graphics/Graphics.cppm
        Graphics/Models/Models.cppm
        Graphics/Models/Dto.cppm
        Graphics/Models/Fbo.cppm
        Graphics/Models/Material.cppm
        Graphics/Models/Mesh.cppm
        Graphics/Models/Scene.cppm
        Graphics/Models/Shader.cppm
        Graphics/Models/Texture.cppm
        Graphics/System/Window.cppm
        Graphics/Api/GraphicsApi.cppm
        Graphics/Api/IRenderer.cppm
        Graphics/Api/VulkanRenderer.cppm
        Graphics/Api/OpenGLRenderer.cppm
        Graphics/Services/SceneManagerService.cppm
        Graphics/Services/RenderableEntityService.cppm
        Graphics/Services/FboService.cppm
        Graphics/Services/ShaderService.cppm
        Graphics/Services/CubeMapService.cppm
        Graphics/Services/PresenterService.cppm
        Graphics/Services/CameraService.cppm
        Graphics/Services/PostProcessService.cppm
        Graphics/Services/SceneService.cppm
        Io/Io.cppm
        Io/Services/GLTFService.cppm
        Io/Services/ResourceService.cppm
        Io/Utility/AccessorUtility.cppm
        Shared/Shared.cppm
        Shared/Types/Resource.cppm)

target_include_directories(Engine SYSTEM PUBLIC
        ${STB_INCLUDE_DIRS}
        ${TINYGLTF_INCLUDE_DIRS}
        ${ozz_animation_SOURCE_DIR}/include)

target_compile_definitions(Engine PUBLIC GLM_ENABLE_EXPERIMENTAL)

target_link_libraries(Engine PUBLIC spdlog::spdlog)
target_link_libraries(Engine PUBLIC glm::glm)
target_link_libraries(Engine PUBLIC glfw)
target_link_libraries(Engine PUBLIC Vulkan::Vulkan)
target_link_libraries(Engine PRIVATE GPUOpen::VulkanMemoryAllocator)
target_link_libraries(Engine PRIVATE unofficial::shaderc::shaderc)
target_link_libraries(Engine PUBLIC ozz_animation ozz_base)
target_link_libraries(Engine PRIVATE glad)
if(WIN32)
    target_link_libraries(Engine PRIVATE opengl32)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(Engine PRIVATE OpenGL::GL)
endif()

target_link_libraries(Engine PRIVATE raceengine_warnings)
