cmake_minimum_required (VERSION 3.8)

add_library (Shared
        Services/MemoryStorageService.cpp
        Services/MemoryStorageService.h)

target_include_directories(Shared PUBLIC ../Graphics)
target_include_directories(Shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/ozz-animation/include)

target_link_libraries(Shared PRIVATE spdlog)
target_link_libraries(Shared PRIVATE glm)

