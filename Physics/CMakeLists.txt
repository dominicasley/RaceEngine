set(CMAKE_CXX_STANDARD 23)
cmake_minimum_required (VERSION 3.28)

add_library(Physics
        Services/PhysicsService.ixx)

target_link_libraries(Physics PUBLIC Shared)
target_link_libraries(Physics PUBLIC glm)
target_link_libraries(Physics PUBLIC spdlog)
target_link_libraries(Physics PRIVATE PhysXExtensions_64 PhysX_64 PhysXVehicle_64 PhysXCharacterKinematic_64 PhysXCooking_64 PhysXCommon_64 PhysXFoundation_64)

target_include_directories(Physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Shared)
target_include_directories(Physics PUBLIC ../vendor/physx/physx/install/vc17win64/physx/include)

target_link_directories(Physics PUBLIC ../vendor/physx/physx/install/vc17win64/physx/bin/win.x86_64.vc143.md/release)
