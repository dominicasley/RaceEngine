set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required (VERSION 3.8)

add_library (Io
        Services/ResourceService.cpp
        Services/ResourceService.h
        Services/GLTFService.cpp
        Services/GLTFService.h
        Utility/AccessorUtility.h)

target_include_directories(Io PUBLIC "../")
target_include_directories(Io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/tinygltf)
target_include_directories(Io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/ozz-animation/include)
include_directories(../vendor/FreeImage/bin)

target_link_libraries(Io PRIVATE spdlog)
target_link_libraries(Io PUBLIC Async)
target_link_libraries(Io PUBLIC glm)
target_link_libraries(Io PUBLIC ozz_animation)
target_link_libraries(Io PUBLIC ozz_base)

link_directories(../vendor/FreeImage/bin)
link_libraries(FreeImage)
