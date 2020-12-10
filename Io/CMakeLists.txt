cmake_minimum_required (VERSION 3.8)

add_library (Io
        "Services/MemoryStorageService.cpp"
        "Services/MemoryStorageService.h"
        "Services/ResourceService.cpp"
        "Services/ResourceService.h"
        Utility/AccessorUtility.h)


target_include_directories(Io PUBLIC "../Async/")
target_include_directories(Io PUBLIC "../Graphics/")
target_include_directories(Io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/tinygltf)
target_include_directories(Io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/ozz-animation/include)

target_link_libraries(Io PRIVATE spdlog)
target_link_libraries(Io PUBLIC Async)
target_link_libraries(Io PUBLIC ozz_animation)
target_link_libraries(Io PUBLIC ozz_base)
