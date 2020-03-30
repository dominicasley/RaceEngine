cmake_minimum_required (VERSION 3.8)

add_library (Io
        "Services/MemoryStorageService.cpp"
        "Services/MemoryStorageService.h"
        "Services/ResourceService.cpp"
        "Services/ResourceService.h")


target_include_directories(Io PUBLIC "../../Middleware/")
target_include_directories(Io PUBLIC "../Async/")
target_include_directories(Io PUBLIC "../../Middleware/tinygltf")
target_link_directories(Io PRIVATE "../../Middleware/spdlog/lib")
target_link_libraries(Io PRIVATE spdlog)
target_link_libraries(Io PUBLIC Async)

