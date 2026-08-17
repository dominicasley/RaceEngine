set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required (VERSION 3.28)

# Warning/sanitizer flags ride an INTERFACE carrier so vendored code stays untouched.
# -Werror arrives once the first-party warning count reaches zero.
add_library(raceengine_warnings INTERFACE)
target_compile_options(raceengine_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion -Wimplicit-fallthrough)
option(RACEENGINE_SANITIZE "Build first-party code with ASan+UBSan" OFF)
if(RACEENGINE_SANITIZE)
    target_compile_options(raceengine_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(raceengine_warnings INTERFACE -fsanitize=address,undefined)
endif()

add_library (Engine
        "Engine.h" "Engine.cpp"
        "Bootstrapper.cpp" "Bootstrapper.h")

target_include_directories(Engine PUBLIC "vendor/di/include")

if (NOT TARGET spdlog)
    add_subdirectory(vendor/spdlog)
endif()

if(NOT PYTHON_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vendor/gl3w)
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vendor/gl3w/gl3w_gen.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vendor/gl3w)
add_subdirectory(vendor/gl3w vendor/gl3w)
add_subdirectory(vendor/glm vendor/glm)
add_subdirectory(vendor/glfw vendor/glfw)

set(ozz_build_samples OFF)
set(ozz_build_tests OFF)
set(ozz_build_howtos OFF)
set(ozz_build_gltf OFF)
add_subdirectory(vendor/ozz-animation vendor/ozz-animation)

if(NOT WIN32)
    add_subdirectory(vendor/FreeImage vendor/FreeImage)
endif()

target_link_libraries(Engine PUBLIC spdlog)
if(WIN32)
    target_link_libraries(Engine PRIVATE opengl32)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(Engine PRIVATE OpenGL::GL)
endif()
target_link_libraries(Engine PUBLIC glm)
target_link_libraries(Engine PRIVATE gl3w)
target_link_libraries(Engine PUBLIC glfw)

if(WIN32)
    target_include_directories(Engine PUBLIC "C:/VulkanSDK/1.2.131.2/Include")
    target_link_directories(Engine PUBLIC "C:/VulkanSDK/1.2.131.2/Lib")
else()
    find_package(Vulkan REQUIRED)
    target_link_libraries(Engine PUBLIC Vulkan::Vulkan)
endif()
target_include_directories(Engine PRIVATE vendor/tinygltf)

project ("Graphics")
add_subdirectory("Graphics")

project ("Io")
add_subdirectory("Io")

project ("Async")
add_subdirectory("Async")

project ("Shared")
add_subdirectory("Shared")

project ("Game")
add_subdirectory("Game")

target_link_libraries(Engine PUBLIC Game)
target_link_libraries(Engine PUBLIC Graphics)
target_link_libraries(Engine PUBLIC Io)
target_link_libraries(Engine PUBLIC Async)
target_link_libraries(Engine PUBLIC Shared)

foreach(firstparty Engine Game Graphics Io Async Shared)
    target_link_libraries(${firstparty} PRIVATE raceengine_warnings)
endforeach()
