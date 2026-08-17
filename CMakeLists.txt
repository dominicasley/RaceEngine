set(CMAKE_CXX_STANDARD 23)
cmake_minimum_required (VERSION 3.28)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

string(
        APPEND _opts
        "$<IF:$<CXX_COMPILER_ID:MSVC>,"
        "/W4;$<$<CONFIG:RELEASE>:/O2>,"
        "-Wall;-Wextra;-Werror;"
        "$<$<CONFIG:RELEASE>:-O3>"
        "$<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>"
        ">"
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/FreeImage/bin/FreeImage.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../OpenStreetRace/bin/)

add_library (Engine
        "Engine.h" "Engine.cpp"
        "Bootstrapper.cpp" "Bootstrapper.h")

target_include_directories(Engine PUBLIC "vendor/di/include")

if (NOT TARGET spdlog)
    add_subdirectory(vendor/spdlog)
endif()

add_subdirectory(vendor/gl3w vendor/gl3w)
add_subdirectory(vendor/glm vendor/glm)
add_subdirectory(vendor/glfw vendor/glfw)

set(ozz_build_samples OFF)
set(ozz_build_tests OFF)
set(ozz_build_howtos OFF)
set(ozz_build_gltf OFF)
add_subdirectory(vendor/ozz-animation vendor/ozz-animation)

target_link_libraries(Engine PUBLIC spdlog)
target_link_libraries(Engine PRIVATE opengl32)
target_link_libraries(Engine PUBLIC glm)
target_link_libraries(Engine PRIVATE gl3w)
target_link_libraries(Engine PUBLIC glfw)

target_include_directories(Engine PUBLIC "C:/VulkanSDK/1.2.131.2/Include")
target_link_directories(Engine PUBLIC "C:/VulkanSDK/1.2.131.2/Lib")
target_include_directories(Engine PRIVATE vendor/tinygltf)
target_link_directories(Engine PUBLIC vendor/physx/physx/bin/win.x86_64.vc143.mt/release)

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

project ("Physics")
add_subdirectory("Physics")

target_link_libraries(Engine PUBLIC Game)
target_link_libraries(Engine PUBLIC Physics)
target_link_libraries(Engine PUBLIC Graphics)
target_link_libraries(Engine PUBLIC Io)
target_link_libraries(Engine PUBLIC Async)
target_link_libraries(Engine PUBLIC Shared)
