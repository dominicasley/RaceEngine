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

# Module *implementation* units are plain sources, deliberately not in the CXX_MODULES file set
# below: that file set is module interfaces, and an interface is what produces an importable BMI.
# These produce an object and nothing else, which is the whole reason they exist — see
# Physics/Impl/TyreImpl.cpp and docs/build-times.md. CMake still scans them, because a unit saying
# `module raceengine.physics;` has to be ordered after that module's interface.
add_library (Engine STATIC
        Io/ThirdPartyImpl.cpp
        Physics/Impl/ClutchImpl.cpp
        Physics/Impl/ContactImpl.cpp
        Physics/Impl/ContactPatchImpl.cpp
        Physics/Impl/CouplingImpl.cpp
        Physics/Impl/DrivelineImpl.cpp
        Physics/Impl/JoltRuntimeImpl.cpp
        Physics/Impl/PhysicsWorldImpl.cpp
        Physics/Impl/ProvingGroundImpl.cpp
        Physics/Impl/PublishedCarsImpl.cpp
        Physics/Impl/RigidBodyImpl.cpp
        Physics/Impl/SetupFileImpl.cpp
        Physics/Impl/SuspensionImpl.cpp
        Physics/Impl/TelemetryImpl.cpp
        Physics/Impl/TyreImpl.cpp
        Physics/Impl/VehicleImpl.cpp)

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
        Graphics/Api/Frustum.cppm
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
        Physics/Api/SetupFile.cppm
        Audio/Audio.cppm
        Audio/Api/CarAudio.cppm
        Audio/Api/EngineLayers.cppm
        Audio/Api/TyreLayers.cppm
        Audio/Api/SoundBank.cppm
        Audio/Api/AudioBackend.cppm
        Audio/Backend/SilentAudioBackend.cppm
        Audio/Backend/FmodAudioBackend.cppm
        Audio/Services/AudioService.cppm
        Io/Io.cppm
        Io/Services/GLTFService.cppm
        Io/Services/ResourceService.cppm
        Io/Utility/AccessorUtility.cppm # module raceengine.io.accessor, not a partition of raceengine.io
        Input/Input.cppm
        Input/Api/DriverInput.cppm
        Input/Api/InputBackend.cppm
        Input/Api/DeviceProfile.cppm
        Input/Api/InputMapping.cppm
        Input/Api/RackTorque.cppm
        Input/Api/ForceMapping.cppm
        Input/Api/PedalFeedback.cppm
        Input/Api/PedalMotors.cppm
        Input/Api/DirectInputContract.cppm
        Input/Api/InputBackendFactory.cppm
        # Implementation partitions, on the renderer's rule: both are built on both platforms so
        # neither rots, and each one's system calls are the only part of it that is conditional.
        Input/Api/InputBackendFactoryImpl.cppm
        Input/Backend/EvdevInputBackend.cppm
        Input/Backend/DirectInputBackend.cppm
        Input/Services/InputService.cppm
        Input/Services/ForceFeedbackService.cppm
        Shared/Shared.cppm
        Shared/Types/Resource.cppm)

target_include_directories(Engine SYSTEM PUBLIC
        ${STB_INCLUDE_DIRS}
        ${TINYGLTF_INCLUDE_DIRS}
        ${ozz_animation_SOURCE_DIR}/include)

# No precompiled header on this target, and that is a finding rather than an omission.
#
# A PCH is worth an enormous amount here in principle — a std header textually included in a global
# module fragment has to be merged against the copy already inside every imported BMI, and in a unit
# that imports `raceengine` that merge is most of the compile. Measured on a probe unit, best of
# three: 11.13 s without a PCH, 1.74 s with one, the unit's own includes untouched.
#
# **clang-19 segfaults building this target's module units with one** (exit 139 in the frontend,
# first on Graphics/Models/Scene.cppm). PCH plus C++20 module interfaces is not stable in this
# compiler, and it fails as a crash rather than as a diagnostic, so there is nothing to work around.
# It is applied to `EngineTests` instead, where the sources are plain translation units and it does
# not crash — see Tests/CMakeLists.txt. Retest on clang-20+; the header is kept ready at
# cmake/raceengine_pch.hpp. Full account: docs/build-times.md.

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

# FMOD, when there is one. Off by default and named rather than discovered: an audio SDK that has to
# be downloaded behind a login is not something a configure step should go looking for, and a build
# that quietly found one on one machine and not another is a build whose sound is a property of the
# machine. `FMOD_ROOT` is the extracted FMOD Engine folder — the one holding `api/`.
option(RACEENGINE_WITH_FMOD "Build the FMOD audio backend" OFF)

if(RACEENGINE_WITH_FMOD)
    if(NOT DEFINED FMOD_ROOT)
        message(FATAL_ERROR "RACEENGINE_WITH_FMOD is ON but FMOD_ROOT is not set")
    endif()

    find_library(FMOD_LIBRARY NAMES fmod PATHS "${FMOD_ROOT}/api/core/lib/x86_64" REQUIRED)
    find_library(FMOD_STUDIO_LIBRARY NAMES fmodstudio PATHS "${FMOD_ROOT}/api/studio/lib/x86_64" REQUIRED)

    # SYSTEM, so FMOD's own headers are not held to this project's warnings. The zero-suppressions
    # rule is first-party only and this is how that is kept: no #pragma, no -Wno-, and -Werror still
    # absolute over every line anybody here wrote. `fmod.hpp` alone trips -Wold-style-cast.
    target_include_directories(Engine SYSTEM PRIVATE
            "${FMOD_ROOT}/api/core/inc"
            "${FMOD_ROOT}/api/studio/inc")
    target_link_libraries(Engine PRIVATE "${FMOD_LIBRARY}" "${FMOD_STUDIO_LIBRARY}")
    target_compile_definitions(Engine PRIVATE RACEENGINE_HAS_FMOD)

    message(STATUS "FMOD: ${FMOD_LIBRARY}")
endif()

# The calibration tool. A console program, no window and no Vulkan: it talks to the input backend
# directly because what it needs is the raw device code, and the service's whole job is to have
# shaped it already. Built as part of `all` so it is always beside the game it writes a profile for.
add_executable(RaceEngineCalibrate Tools/Calibrate.cpp)
target_link_libraries(RaceEngineCalibrate PRIVATE Engine)
target_link_libraries(RaceEngineCalibrate PRIVATE raceengine_warnings)
set_target_properties(RaceEngineCalibrate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
