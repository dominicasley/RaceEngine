# The engine's unit tests. Nothing ctest runs opens a device, a window or a display: the services
# under test take their seams by reference, so a recording fake stands in for the backend and the
# tests run anywhere. The GPU gates are the shell scripts in scripts/, and they are untouched by
# this.
find_package(Catch2 3 CONFIG REQUIRED)

add_executable(EngineTests
        AccessorUtilityTests.cpp
        CameraServiceTests.cpp
        ClutchTests.cpp
        ContactPatchTests.cpp
        CouplingTests.cpp
        DrivelineTests.cpp
        FrameDiagnosticsTests.cpp
        GltfColourSpaceTests.cpp
        GltfNodeTransformTests.cpp
        GolfGtiTests.cpp
        HandlingTests.cpp
        JoltRuntimeTests.cpp
        LookupTableTests.cpp
        MemoryStorageTests.cpp
        PhysicalCameraTests.cpp
        PhysicsWorldTests.cpp
        PostProcessingTests.cpp
        ProvingGroundTests.cpp
        PublishedGeometryTests.cpp
        RendererSeamTests.cpp
        RigidBodyTests.cpp
        SceneManagerServiceTests.cpp
        ShiftTests.cpp
        SuspensionTests.cpp
        TelemetryTests.cpp
        TorqueConverterTests.cpp
        TyreTests.cpp
        VehicleTests.cpp
        SphericalHarmonicsTests.cpp
        ShadowCascadesTests.cpp)

# The recording fake is a module unit rather than a header because the seams it implements are
# only reachable through `import`, and a header may not import.
target_sources(EngineTests PRIVATE
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES CapturedLog.cppm RecordingRenderBackend.cppm)

target_link_libraries(EngineTests PRIVATE Engine)
target_link_libraries(EngineTests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(EngineTests PRIVATE raceengine_warnings)

# One CTest entry per TEST_CASE, so a failure names the case rather than the binary. The
# enumeration runs the freshly built binary, which is itself a check that it starts headless.
include(Catch)
catch_discover_tests(EngineTests)