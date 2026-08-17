set(CMAKE_CXX_STANDARD 23)
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

find_package(spdlog CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
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
        "Engine.h" "Engine.cpp"
        "Bootstrapper.cpp" "Bootstrapper.h"
        Graphics/Api/VulkanRenderer.cpp
        Graphics/Api/VulkanRenderer.h
        Graphics/Api/OpenGLRenderer.cpp
        Graphics/Api/OpenGLRenderer.h
        Graphics/Models/Scene/CreateRenderableModelDTO.h
        Graphics/Models/Scene/ShaderDescriptor.h
        Graphics/Models/Scene/Camera.h
        Graphics/Models/Scene/RenderableEntity.h
        Graphics/Models/Scene/Light.h
        Graphics/Models/Scene/Scene.h
        Graphics/Services/SceneManagerService.cpp
        Graphics/Services/SceneManagerService.h
        Graphics/System/GlfwWindow.cpp
        Graphics/System/GlfwWindow.h
        Graphics/System/IWindow.h
        Graphics/Api/VulkanWindowRequiredExtensions.h
        Graphics/System/WindowState.h
        Graphics/Services/RenderableEntityService.cpp
        Graphics/Services/RenderableEntityService.h
        Graphics/Services/SceneService.cpp
        Graphics/Services/SceneService.h
        Graphics/Services/CameraService.cpp
        Graphics/Services/CameraService.h
        Graphics/Services/AnimationService.cpp
        Graphics/Services/AnimationService.h
        Graphics/Models/Scene/Model.h
        Graphics/Models/Scene/Mesh.h
        Graphics/Models/Scene/MeshBuffer.h
        Graphics/Models/Scene/MeshPrimitive.h
        Graphics/Models/Scene/PrimitiveAttributeType.h
        Graphics/Models/Scene/Material.h
        Graphics/Models/Scene/Texture.h
        Graphics/Models/Scene/TextureFormat.h
        Graphics/Models/Scene/PixelDataType.h
        Graphics/Models/Scene/VertexIndicesType.h
        Graphics/Models/Scene/SceneNode.h
        Graphics/Models/Scene/RenderableMesh.h
        Graphics/Services/ShaderService.cpp
        Graphics/Services/ShaderService.h
        Graphics/Models/Scene/Shader.h
        Graphics/Models/Scene/CubeMap.h
        Graphics/Services/CubeMapService.cpp
        Graphics/Services/CubeMapService.h
        Graphics/Models/Scene/RenderableEntityType.h
        Graphics/Models/Scene/RenderableModel.h
        Graphics/Models/Scene/Fbo.h
        Graphics/Models/Scene/FboAttachment.h
        Graphics/Models/Scene/FboAttachmentType.h
        Graphics/Models/Scene/FboType.h
        Graphics/Services/FboService.cpp
        Graphics/Services/FboService.h
        Graphics/Models/Scene/CreateFboDTO.h
        Graphics/Models/Scene/CreateFboAttachmentDTO.h
        Graphics/Models/Scene/PostProcess.h
        Graphics/Services/PostProcessService.cpp
        Graphics/Services/PostProcessService.h
        Graphics/Services/PresenterService.cpp
        Graphics/Services/PresenterService.h
        Graphics/Models/Scene/Presenter.h
        Io/Services/ResourceService.cpp
        Io/Services/ResourceService.h
        Io/Services/GLTFService.cpp
        Io/Services/GLTFService.h
        Io/Utility/AccessorUtility.h
        Async/Models/IAsyncTask.h
        Async/Models/Observable.h
        Async/Services/BackgroundWorkerService.cpp
        Async/Services/BackgroundWorkerService.h
        Async/Operators/ForkJoin.h
        Async/Models/AsyncTask.h
        Async/Models/ITask.h
        Shared/Services/MemoryStorageService.cpp
        Shared/Services/MemoryStorageService.h
        Shared/Types/Resource.h
        Game/Components/Drawable.h
        Game/Models/Component.h
        Game/Models/Entity.h
        Game/Services/EntityService.cpp
        Game/Services/EntityService.h)

target_include_directories(Engine PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Shared)
target_include_directories(Engine SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/di-header
        ${STB_INCLUDE_DIRS}
        ${TINYGLTF_INCLUDE_DIRS}
        ${ozz_animation_SOURCE_DIR}/include)

target_compile_definitions(Engine PUBLIC GLM_ENABLE_EXPERIMENTAL)

target_link_libraries(Engine PUBLIC spdlog::spdlog)
target_link_libraries(Engine PUBLIC glm::glm)
target_link_libraries(Engine PUBLIC glfw)
target_link_libraries(Engine PUBLIC Vulkan::Vulkan)
target_link_libraries(Engine PUBLIC ozz_animation ozz_base)
target_link_libraries(Engine PRIVATE glad)
if(WIN32)
    target_link_libraries(Engine PRIVATE opengl32)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(Engine PRIVATE OpenGL::GL)
endif()

target_link_libraries(Engine PRIVATE raceengine_warnings)
