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
# No features requested, and that is the safe answer rather than a lazy one: the port's debug
# renderer, profiler and C++ RTTI options each change JPH_ prefixed preprocessor state that alters
# the layout of Jolt's own types, so a consumer compiled against a different set than the library
# was built with mismatches silently. The imported target propagates exactly what it was built
# with, which is what keeps our translation units and the archive agreeing.
find_package(Jolt CONFIG REQUIRED)
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

# The physics backend is its own plain (non-module) target, and that is load-bearing rather than
# tidy. Jolt::Jolt carries `-mavx2 -mfma -mbmi -mlzcnt -mf16c -mpopcnt` and eight JPH_USE_* defines
# as usage requirements, and a clang BMI records the target features it was built with — so linking
# Jolt into Engine makes every translation unit that *imports* an Engine module fail to load the
# BMI unless it is built with Jolt's instruction set too. That is the tests and the sandbox, neither
# of which has anything to do with physics. Confining Jolt here keeps its usage requirements off
# Engine's translation units, off Engine's BMIs, and off everything downstream of them, and leaves
# the engine's own codegen exactly as it was.
add_library(EnginePhysicsBackend STATIC Physics/Backend/JoltBackend.cpp)
target_link_libraries(EnginePhysicsBackend PRIVATE Jolt::Jolt)
target_link_libraries(EnginePhysicsBackend PRIVATE raceengine_warnings)

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
        Graphics/Models/LightProbe.cppm
        Graphics/Models/Material.cppm
        Graphics/Models/Mesh.cppm
        Graphics/Models/Scene.cppm
        Graphics/Models/Shader.cppm
        Graphics/Models/Texture.cppm
        Graphics/System/Window.cppm
        Graphics/Api/RenderContract.cppm
        Graphics/Api/SphericalHarmonics.cppm
        Graphics/Api/ShadowCascades.cppm
        Graphics/Api/PostProcessing.cppm
        Graphics/Api/LookupTable.cppm
        Graphics/Api/PhysicalCamera.cppm
        Graphics/Api/FrameDiagnostics.cppm
        Graphics/Api/IFrameRecorder.cppm
        Graphics/Api/IGpuResourceFactory.cppm
        Graphics/Api/IFrameCapture.cppm
        Graphics/Api/IRenderBackend.cppm
        Graphics/Api/RenderBackendFactory.cppm
        # Implementation partitions: they produce BMIs like any other module unit, so CMake
        # needs them in the module file set, but nothing outside raceengine.graphics can import
        # them and no consumer's module map names them.
        Graphics/Api/RenderBackendFactoryImpl.cppm
        Graphics/Api/VulkanRenderer.cppm
        Graphics/Services/SceneManagerService.cppm
        Graphics/Services/RenderableEntityService.cppm
        Graphics/Services/FboService.cppm
        Graphics/Services/ShaderService.cppm
        Graphics/Services/CubeMapService.cppm
        Graphics/Services/LightProbeService.cppm
        Graphics/Services/PresenterService.cppm
        Graphics/Services/CameraService.cppm
        Graphics/Services/PostProcessService.cppm
        Graphics/Services/AutoExposureService.cppm
        Graphics/Services/AmbientOcclusionService.cppm
        Graphics/Services/BloomService.cppm
        Graphics/Services/ColourGradeService.cppm
        Graphics/Services/SceneService.cppm
        Graphics/Services/ShadowService.cppm
        Graphics/Services/AssetService.cppm
        Physics/Physics.cppm
        Physics/Api/RigidBody.cppm
        Physics/Api/JoltRuntime.cppm
        Physics/Api/ProvingGround.cppm
        Physics/Api/PhysicsWorld.cppm
        Physics/Api/Suspension.cppm
        Physics/Api/ContactPatch.cppm
        Physics/Api/Tyre.cppm
        Physics/Api/Contact.cppm
        Physics/Api/Telemetry.cppm
        Physics/Api/Vehicle.cppm
        Physics/Api/Coupling.cppm
        Physics/Api/Clutch.cppm
        Physics/Api/Driveline.cppm
        Physics/Api/PublishedCars.cppm
        Io/Io.cppm
        Io/Services/GLTFService.cppm
        Io/Services/ResourceService.cppm
        Io/Utility/AccessorUtility.cppm # module raceengine.io.accessor, not a partition of raceengine.io
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

target_link_libraries(Engine PRIVATE EnginePhysicsBackend)

target_link_libraries(Engine PRIVATE raceengine_warnings)

# Unit tests. They construct services directly against fakes and never open a device, a window or
# a display, which is the point: the shell gates in scripts/ need a GPU and these must not.
# Built as part of `all` so that `cmake --build <dir>` followed by `ctest --test-dir <dir>` works
# without a second build step; -DRACEENGINE_BUILD_TESTS=OFF drops the target entirely.
option(RACEENGINE_BUILD_TESTS "Build the RaceEngine unit tests" ON)
if(RACEENGINE_BUILD_TESTS)
    add_subdirectory(Tests)
endif()
