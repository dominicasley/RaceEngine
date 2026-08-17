set(CMAKE_CXX_STANDARD 23)
cmake_minimum_required (VERSION 3.28)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library (Shared
        Services/MemoryStorageService.cpp
        Services/MemoryStorageService.h Types/Resource.h)

target_include_directories(Shared PUBLIC ../)
target_include_directories(Shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/ozz-animation/include)

target_link_libraries(Shared PRIVATE spdlog)
target_link_libraries(Shared PRIVATE glm)

