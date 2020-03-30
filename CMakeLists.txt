cmake_minimum_required (VERSION 3.8)

add_library (Engine
        "Engine.h" "Engine.cpp"
        "Bootstrapper.cpp" "Bootstrapper.h")


project ("Graphics")
add_subdirectory("Graphics")

project ("Io")
add_subdirectory("Io")

project ("Async")
add_subdirectory("Async")

target_link_libraries(Engine PUBLIC Graphics)
target_link_libraries(Engine PUBLIC Io)
target_link_libraries(Engine PUBLIC Async)

target_include_directories(Engine PUBLIC "../Middleware")
target_link_directories(Engine PUBLIC "../Middleware/spdlog/lib")
target_link_libraries(Engine PRIVATE spdlog)

