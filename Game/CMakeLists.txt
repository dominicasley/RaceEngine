set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required (VERSION 3.8)

add_library (Game
        Components/Drawable.h
        Models/Component.h
        Models/Entity.h
        Services/EntityService.cpp
        Services/EntityService.h)

target_include_directories(Game PUBLIC "../")

target_link_libraries(Game PUBLIC Shared)
target_include_directories(Game PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Shared)

target_link_libraries(Game PUBLIC Graphics)
target_link_libraries(Game PUBLIC spdlog)



