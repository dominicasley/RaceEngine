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
        "Api/VulkanWindowRequiredExtensions.h")

target_include_directories(Graphics PUBLIC "C:/VulkanSDK/1.2.131.2/Include")
target_include_directories(Graphics PUBLIC "../../Middleware/glfw/include/")
target_include_directories(Graphics PUBLIC "../../Middleware/glm")
target_include_directories(Graphics PUBLIC "../../Middleware/FreeImage")
target_include_directories(Graphics PUBLIC "../../Middleware/tinygltf")
target_include_directories(Graphics PUBLIC "../../Middleware/")
target_link_directories(Graphics PUBLIC "C:/VulkanSDK/1.2.131.2/Lib")
target_link_directories(Graphics PUBLIC "../../Middleware/glfw/lib-vc2019/")
target_link_directories(Graphics PUBLIC "../../Middleware/FreeImage")
target_link_directories(Graphics PRIVATE "../../Middleware/spdlog/lib")
target_link_directories(Graphics PUBLIC "../../Middleware/glew/lib")
target_link_libraries(Graphics PRIVATE opengl32)
target_link_libraries(Graphics PRIVATE FreeImage)
target_link_libraries(Graphics PRIVATE glew32)
target_link_libraries(Graphics PRIVATE spdlog)
target_link_libraries(Graphics PRIVATE dwmapi)

# TODO: Add tests and install targets if needed.
