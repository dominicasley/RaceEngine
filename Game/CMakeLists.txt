set(CMAKE_CXX_STANDARD 23)
cmake_minimum_required (VERSION 3.28)

add_library (Game
        Game.ixx
        Components.ixx
        Models.ixx
        Services.ixx)

target_include_directories(Game PUBLIC "../")

target_link_libraries(Game PUBLIC Shared)
target_include_directories(Game PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Shared)

target_link_libraries(Game PUBLIC Graphics)
target_link_libraries(Game PUBLIC spdlog)



