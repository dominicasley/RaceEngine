# CMakeList.txt : CMake project for OpenStreetRace, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_library (Graphics
        "Api/VulkanRenderer.cpp"
        "Api/VulkanRenderer.h"
        "Api/OpenGLRenderer.cpp"
        "Api/OpenGLRenderer.h"
        "Api/Resources/Animation.h"
        "Api/Resources/Material.cpp"
        "Api/Resources/Material.h"
        "Api/Resources/Mesh.h"
        "Api/Resources/Model.cpp"
        "Api/Resources/Model.h"
        "Api/Resources/RenderableEntityDesc.h"
        "Api/Resources/Shader.h"
        "Api/Resources/Skeleton.cpp"
        "Api/Resources/Skeleton.h"
        "Api/Resources/Texture.cpp"
        "Api/Resources/Texture.h"
        "Api/Resources/Uniform.h"
        "Api/Resources/UniformDataType.h"
        "Scene/Camera.cpp"
        "Scene/Camera.h"
        "Scene/RenderableEntity.cpp"
        "Scene/RenderableEntity.h"
        "Scene/Light.cpp"
        "Scene/Light.h"
        "Scene/Scene.cpp"
        "Scene/Scene.h"
        "Scene/SceneManager.cpp"
        "Scene/SceneManager.h"
        "System/GlfwWindow.cpp"
        "System/GlfwWindow.h"
        "System/IWindow.h"
        "Api/VulkanWindowRequiredExtensions.h"
        )

target_include_directories(Graphics PUBLIC "C:/VulkanSDK/1.2.131.2/Include")
target_link_directories(Graphics PUBLIC "C:/VulkanSDK/1.2.131.2/Lib")

target_link_libraries(Graphics PUBLIC opengl32)
target_link_libraries(Graphics PUBLIC spdlog)
target_link_libraries(Graphics PUBLIC glm)
target_link_libraries(Graphics PUBLIC gl3w)
target_link_libraries(Graphics PUBLIC glfw)
target_include_directories(Graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/tinygltf)

# TODO: Add tests and install targets if needed.
