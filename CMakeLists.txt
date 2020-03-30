cmake_minimum_required (VERSION 3.8)

add_library (Engine
        "Engine.h" "Engine.cpp"
        "Bootstrapper.cpp" "Bootstrapper.h")

target_include_directories(Engine PUBLIC "vendor/di/include")

if (NOT TARGET spdlog)
    add_subdirectory(vendor/spdlog)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vendor/gl3w/gl3w_gen.py)
add_subdirectory(vendor/gl3w vendor/gl3w)
add_subdirectory(vendor/glm vendor/glm)
add_subdirectory(vendor/glfw vendor/glfw)

target_link_libraries(Engine PUBLIC spdlog)
target_link_libraries(Engine PRIVATE opengl32)
target_link_libraries(Engine PUBLIC glm)
target_link_libraries(Engine PRIVATE gl3w)
target_link_libraries(Engine PUBLIC glfw)

target_include_directories(Engine PUBLIC "C:/VulkanSDK/1.2.131.2/Include")
target_link_directories(Engine PUBLIC "C:/VulkanSDK/1.2.131.2/Lib")
target_include_directories(Engine PUBLIC vendor/tinygltf)


project ("Graphics")
add_subdirectory("Graphics")

project ("Io")
add_subdirectory("Io")

project ("Async")
add_subdirectory("Async")

target_link_libraries(Engine PUBLIC Graphics)
target_link_libraries(Engine PUBLIC Io)
target_link_libraries(Engine PUBLIC Async)


