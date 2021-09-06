# CMakeList.txt : CMake project for OpenStreetRace, include source and define
# project specific logic here.
#
set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_library (Graphics
        Api/VulkanRenderer.cpp
        Api/VulkanRenderer.h
        Api/OpenGLRenderer.cpp
        Api/OpenGLRenderer.h
        Models/Scene/CreateRenderableModelDTO.h
        Models/Scene/ShaderDescriptor.h
        Models/Scene/Skeleton.h
        Api/Resources/Uniform.h
        Api/Resources/UniformDataType.h
        Models/Scene/Camera.h
        Models/Scene/RenderableEntity.h
        Models/Scene/Light.h
        Models/Scene/Scene.h
        Services/SceneManagerService.cpp
        Services/SceneManagerService.h
        System/GlfwWindow.cpp
        System/GlfwWindow.h
        System/IWindow.h
        Api/VulkanWindowRequiredExtensions.h
        System/WindowState.h
        Services/RenderableEntityService.cpp
        Services/RenderableEntityService.h
        Services/SceneService.cpp
        Services/SceneService.h
        Services/CameraService.cpp
        Services/CameraService.h
        Models/Scene/Joint.h
        Services/AnimationService.cpp
        Services/AnimationService.h
        Models/Scene/Model.h
        Models/Scene/Mesh.h
        Models/Scene/Skin.h
        Models/Scene/MeshBuffer.h
        Models/Scene/MeshPrimitive.h
        Models/Scene/PrimitiveAttributeType.h
        Models/Scene/Material.h
        Models/Scene/Texture.h
        Models/Scene/TextureFormat.h
        Models/Scene/PixelDataType.h
        Models/Scene/VertexIndicesType.h
        Models/Scene/SceneNode.h
        Models/Scene/RenderableMesh.h
        Services/ShaderService.cpp
        Services/ShaderService.h
        Models/Scene/Shader.h
        Models/Scene/CubeMap.h
        Services/CubeMapService.cpp
        Services/CubeMapService.h
        Models/Scene/RenderableEntityType.h
        Models/Scene/RenderableModel.h
        Models/Scene/RenderableSkybox.h
        Models/Scene/Fbo.h
        Models/Scene/FboAttachment.h
        Models/Scene/FboAttachmentType.h
        Models/Scene/FboType.h
        Services/FboService.cpp
        Services/FboService.h
        Models/Scene/CreateFboDTO.h
        Models/Scene/CreateFboAttachmentDTO.h
        Models/Scene/PostProcess.h
        Services/PostProcessService.cpp
        Services/PostProcessService.h
        Services/PresenterService.cpp
        Services/PresenterService.h
        Models/Scene/Presenter.h)

target_include_directories(Graphics PUBLIC "../")

target_include_directories(Graphics PUBLIC "C:/VulkanSDK/1.2.131.2/Include")
target_link_directories(Graphics PUBLIC "C:/VulkanSDK/1.2.131.2/Lib")

target_link_libraries(Graphics PUBLIC opengl32)
target_link_libraries(Graphics PUBLIC spdlog)
target_link_libraries(Graphics PUBLIC glm)
target_link_libraries(Graphics PUBLIC gl3w)
target_link_libraries(Graphics PUBLIC glfw)
target_link_libraries(Graphics PUBLIC ozz_animation)
target_link_libraries(Graphics PUBLIC ozz_base)

target_link_libraries(Graphics PUBLIC Shared)
target_include_directories(Graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Shared)

target_include_directories(Graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/tinygltf)
target_include_directories(Graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/ozz-animation/include)


# TODO: Add tests and install targets if needed.
